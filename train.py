import megengine
import megengine.functional as F
import megengine.module as M
import megengine.optimizer as optim
import megengine.autodiff as autodiff
import megengine.distributed as dist

import utils
from clearml import Task
import os
from tensorboardX import SummaryWriter
from loss import zipf_loss
import argparse
from datasets import load_dataset
import models
logging = megengine.logger.get_logger()

def train(optimizer, dataset, base_lr, train_dataloader, gm, model, train_writer, epoch, nums_epoch, loss_lambda, alpha, dense_rank):
    model.train()
    lr = utils.adjust_learning_rate(dataset, optimizer, epoch, base_lr, nums_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    training_loss = 0
    nums_train_correct, nums_train_example = 0, 0
    minibatch_count = len(train_dataloader)
    for i, data in enumerate(train_dataloader):
        image, labels = data[:2]
        image = megengine.Tensor(image, dtype='float32')
        labels = megengine.Tensor(labels, dtype='int32')
        with gm:
            outs = model(image)
            logits, feats, _ = outs[0]
            dense_logits_list = []
            for layer_out in outs:
                dense_logits_list.append(layer_out[2])
            dense_logits = F.concat(dense_logits_list, axis=1)
            kd_loss = zipf_loss(logits, dense_logits, labels, loss_mask=False, dense=dense_rank)

            xent_alphas = [1] + [alpha]*(len(outs) - 1)
            xent_losses = []
            for layer_out in outs:
                _preds = layer_out[0]
                xent_loss = F.nn.cross_entropy(_preds, labels)
                xent_losses.append(xent_loss)

            total_loss = loss_lambda * kd_loss
            for _alpha, xent_loss in zip(xent_alphas, xent_losses):
                total_loss += _alpha * xent_loss


            gm.backward(total_loss)
            optimizer.step().clear_grad()

        training_loss += total_loss.item() * len(image)
        pred = F.argmax(logits, axis=1)
        nums_train_correct += (pred == labels).sum().item()
        nums_train_example += len(image)

        acc1, acc5 = F.nn.topk_accuracy(logits, labels, topk=(1, 5))

        # logs
        step = epoch * minibatch_count + i
        if step % 200 == 0 and dist.get_rank() == 0:
            train_writer.add_scalar('train/acc1', acc1.item() * 100., step)
            train_writer.add_scalar('train/acc5', acc5.item() * 100., step)
            train_writer.add_scalar(
                'train/total_loss', total_loss.item(), step)
            train_writer.add_scalar('train/lr', lr, step)
            train_writer.flush()

            training_acc = nums_train_correct / nums_train_example
            training_loss /= nums_train_example
            logging.info(f'Epoch = {epoch}, '
                    f'train_loss = {training_loss:.3f}, '
                    f'train_acc = {training_acc:.3f}, '
                    f'lr = {lr:.3f}, '
                    )
    
def evaluate(model, test_dataloader, epoch, val_writer, best_acc):
    model.eval()
    acc1_sum = 0
    acc5_sum = 0
    total_loss = 0
    valdation_num = 0
    for data in test_dataloader:
        image, labels = data[:2]
        image = megengine.Tensor(image, dtype='float32')
        labels = megengine.Tensor(labels, dtype='int32')
        logits = model(image)[0][0]
        num = labels.shape[0]
        acc1, acc5 = F.nn.topk_accuracy(logits, labels, topk=(1, 5)) 
        loss = F.nn.cross_entropy(logits, labels)
        # calculate mean values
        if dist.get_world_size() > 1:
            loss = F.distributed.all_reduce_sum(loss) / dist.get_world_size()
            acc1 = F.distributed.all_reduce_sum(acc1) / dist.get_world_size()
            acc5 = F.distributed.all_reduce_sum(acc5) / dist.get_world_size()
        valdation_num += num
        acc1_sum += acc1.item() * num
        acc5_sum += acc5.item() * num
        total_loss += loss.item()

    total_loss = total_loss / len(test_dataloader)
    acc1 = acc1_sum / valdation_num
    acc5 = acc5_sum / valdation_num        

    if best_acc < acc1:
        best_acc = acc1

    if dist.get_rank() == 0:
        val_writer.add_scalar('val/acc1', acc1 * 100., epoch)
        val_writer.add_scalar('val/acc5', acc5 * 100., epoch)
        val_writer.add_scalar('val/loss', total_loss, epoch)
        val_writer.flush()
        logging.info(f'Epoch = {epoch}, '
                    f'val_acc = {acc1:.3f}, '
                    f'best_acc = {best_acc:.3f}, '
                    )
    return best_acc

def worker(args):
    log_dir = os.path.join(args.log_dir, args.dataset, args.arch)
    train_writer, val_writer = None, None
    if dist.get_rank() == 0:
        os.makedirs(log_dir, exist_ok=True)
        megengine.logger.set_log_file(os.path.join(log_dir, "log.txt"))

        train_writer = SummaryWriter(os.path.join(log_dir, 'train.events'))
        val_writer = SummaryWriter(os.path.join(log_dir, 'val.events'))

    train_dataloader, test_dataloader, num_classes = load_dataset(args.dataset, args.data_dir, args.batch_size, args.workers)  

    model = models.load_model(args.arch, num_classes, upsample=args.upsample)
    # Sync parameters and buffers
    if dist.get_world_size() > 1:
        dist.bcast_list_(model.parameters())
        dist.bcast_list_(model.buffers())

    # Autodiff gradient manager
    gm = autodiff.GradManager().attach(
        model.parameters(),
        callbacks=dist.make_allreduce_cb("mean") if dist.get_world_size() > 1 else None,
    )
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

    if dist.get_rank() == 0:
        project_name = 'zipf_prior/megengine/{}/{}'.format(args.dataset, args.arch)
        task = Task.init(project_name=project_name, task_name=args.desc)

    # Training and validation
    best_acc = 0.0
    for epoch in range(args.epochs):
        train(optimizer, args.dataset, args.lr, train_dataloader, gm, model, train_writer, 
            epoch, args.epochs, args.loss_lambda, args.alpha, args.dense)
        best_acc= evaluate(model, test_dataloader, epoch, val_writer, best_acc)

        # save checkpoint
        if dist.get_rank() == 0:
            megengine.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                },
                os.path.join(log_dir, "checkpoint.pkl"),
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='CIFAR100', type=str,
                        help='dataset name')
    parser.add_argument('--resolution', default=32, type=int,
                        help='input size')
    parser.add_argument('--batch_size', '-b', default=128,
                        type=int, help='batch size')
    parser.add_argument('--val_batch_size', default=None,
                        type=int, help='validation batch size')
    parser.add_argument('--epochs', '-e', default=200,
                        type=int, help='stop epoch')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--decay', default=1e-4,
                        type=float, help='weight decay')
    parser.add_argument('--arch', '-a', default='CIFAR_ResNet18', type=str,
                        help='model type')
    parser.add_argument('--mask', action='store_true', default=False)
    # zipf-law
    parser.add_argument('--loss_lambda', type=float, default=0.0)
    parser.add_argument('--distribution', type=str, default='rank_zipf', choices=['rank_zipf', 'random_uniform', 'random_gaussian',
                                                                                      'random_pareto', 'constant', 'unbiased_zipf', 'sorted_pareto', 'linear_decay', 'exponential_decay'])
    parser.add_argument('--dense', action='store_true', default=False)
    parser.add_argument('--power', default=1.0, type=float)
    # multi layer params
    parser.add_argument('--alpha', default=0.0, type=float)
    parser.add_argument('--upsample', default=None, type=str,
                        choices=['bilinear', 'bicubic', 'None'])
    parser.add_argument('--deeplayer', default=1, type=int, choices=[1, 2, 3])
    # end multi layer
    parser.add_argument('--desc', type=str, default='mge.baseline.1434',
                        help='descript your exp')
    parser.add_argument('--run_num', '-r', default='0',
                        type=str, help='running number')
    parser.add_argument('-c', '--continue-train',
                        action='store_true', default=False)
    parser.add_argument('--log_dir',type=str, default='train_log')
    parser.add_argument('--data_dir',type=str, default='/data/PublicDatasets/cifar100')
    parser.add_argument('--workers', default=8, type=int)

    parser.add_argument('-n','--ngpus', default=None, type=int,help='number of GPUs per node (default: None, use all available GPUs)',)
    parser.add_argument('--dist-addr', default='localhost')
    parser.add_argument('--dist-port', default=23456, type=int)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)

    args = parser.parse_args()

    if args.ngpus is None:
        args.ngpus = dist.helper.get_device_count_by_fork('gpu')

    if args.world_size * args.ngpus > 1:
        dist_worker = dist.launcher(
            master_ip=args.dist_addr,
            port=args.dist_port,
            world_size=args.world_size * args.ngpus,
            rank_start=args.rank * args.ngpus,
            n_gpus=args.ngpus
        )(worker)
        dist_worker(args)
    else:
        worker(args)

if __name__ == '__main__':
    main()