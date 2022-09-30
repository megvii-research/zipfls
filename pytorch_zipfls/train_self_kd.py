#!/usr/bin/env python3
import utils
import settings
import datasets
from utils import WorklogLogger, log_rate_limited
from utils import make_symlink_if_not_exists
from tensorboardX import SummaryWriter
from utils import ensure_dir
import torchvision.models as models
from tqdm import tqdm
import torch.nn as nn
import time
import torch
import argparse
import os
# import timm
from clearml import Task
import cifar_models
from losses import bake, hard_friend_rank, soft_friend_rank, zipf_loss
from torch import Tensor
import torch.nn.functional as F

logger = settings.logger


def forward_with_feat_dense(self, x: Tensor) -> Tensor:
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    # get dense
    n, c, h, w = x.size()
    dense_logits = self.fc(x.permute(0, 2, 3, 1).reshape(
        n*h*w, c)).reshape(n, h*w, -1).argmax(axis=2)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    feat = x
    x = self.fc(x)

    return x, feat, dense_logits


setattr(models.__dict__['ResNet'], 'forward', forward_with_feat_dense)


def loss_fn_kd(outputs, labels, teacher_outputs, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """

    alpha = params['alpha']
    T = params['temperature']
    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                                                  F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
        F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


def train(train_loader, net, criterion, optimizer, epoch, train_writer, log_output, loss_lambda, loss_mask, distribution, dense, rank, friend, power, batchsize, teacher_model):
    teacher_model.eval()

    # switch to train mode
    net.train()
    time_epoch_start = tstart = time.time()
    minibatch_count = len(train_loader)
    for i, (images, labels) in enumerate(train_loader):
        # learning_rate = scheduler.optimizer.param_groups[0]['lr']
        learning_rate = optimizer.param_groups[0]['lr']

        # measure data loading time
        tdata = time.time() - tstart

        labels = labels.cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)
        # compute output
        preds, features, dense_logits = net(images)

        kd_params = {'alpha': 0.5, 'temperature': 1.0}
        with torch.no_grad():
            tea_preds, _, _ = teacher_model(images)
            tea_preds = tea_preds.cuda()

        tea_kd_loss = loss_fn_kd(
            preds, labels, tea_preds, kd_params)

        # ****************************************************
        if dense:
            kd_loss = zipf_loss(preds, dense_logits, labels, loss_mask,
                                distribution, dense, rank,  friend, power)
        else:
            kd_loss = zipf_loss(preds, features, labels, loss_mask,
                                distribution, dense, rank,  friend, power)

        # ****************************************************
        loss = criterion(preds, labels)
        total_loss = loss + loss_lambda * kd_loss + loss_lambda * tea_kd_loss

        # measure accuracy and record loss
        acc1, acc5 = utils.accuracy(preds, labels, topk=(1, 5))

        tea_acc1, tea_acc5 = utils.accuracy(tea_preds, labels, topk=(1, 5))

        # compute gradient and do sgd step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        tend = time.time()
        ttrain = tend - tstart
        tstart = tend

        time_passed = tend - time_epoch_start
        time_expected = time_passed / \
            (batchsize + 1) * minibatch_count
        eta = time_expected - time_passed
        outputs = [
            "e:{},{}/{}".format(epoch, i, minibatch_count),
            "{:.2g} mb/s".format(1. / ttrain),
        ] + [
            'passed:{:.2f}'.format(time_passed),
            'eta:{:.2f}'.format(eta),
            'lr:{:.2f}'.format(learning_rate),
        ] + [
            'acc1:{:.4f}'.format(acc1.item()),
            'tea_acc1:{:.4f}'.format(tea_acc1.item()),
            'acc5:{:.4f}'.format(acc5.item()),
            'xent_loss:{:.4f}'.format(loss.item()),
            'kd_loss:{:.4f}'.format(kd_loss.item()),
            'tea_kd_loss:{:.4f}'.format(tea_kd_loss.item()),
            'total_loss:{:.4f}'.format(total_loss.item()),
        ]

        if tdata / ttrain > .05:
            outputs += [
                "dp/tot: {:.2g}".format(tdata / ttrain),
            ]

        step = epoch * minibatch_count + i
        if step % 200 == 0:
            train_writer.add_scalar('train/acc1', acc1.item(), step)
            train_writer.add_scalar('train/tea_acc1', tea_acc1.item(), step)
            train_writer.add_scalar('train/acc5', acc5.item(), step)
            train_writer.add_scalar('train/xent_loss', loss.item(), step)
            train_writer.add_scalar('train/kd_loss', kd_loss.item(), step)
            train_writer.add_scalar(
                'train/tea_kd_loss', tea_kd_loss.item(), step)
            train_writer.add_scalar(
                'train/total_loss', total_loss.item(), step)
            train_writer.add_scalar('train/lr', learning_rate, step)
            train_writer.flush()
        log_output(' '.join(outputs))


def validate(val_loader, net, criterion, epoch, val_writer, log_output):
    # switch to evaluate mode
    logger.info('eval epoch {}'.format(epoch))
    net.eval()

    acc1_sum = 0
    acc5_sum = 0
    loss = 0
    valdation_num = 0

    for i, (images, labels) in tqdm(enumerate(val_loader)):
        # compute output
        preds, _, _ = net(images)
        labels = labels.cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)

        # measure accuracy and record loss
        acc1, acc5 = utils.accuracy(preds, labels, topk=(1, 5))
        num = labels.size(0)
        valdation_num += num
        acc1_sum += acc1.item() * num
        acc5_sum += acc5.item() * num
        loss += criterion(preds, labels).item()

    loss = loss / len(val_loader)
    acc1 = acc1_sum / valdation_num
    acc5 = acc5_sum / valdation_num

    outputs = [
        "val e:{}".format(epoch),
        'acc1:{:.4f}'.format(acc1),
        'acc5:{:.4f}'.format(acc5),
        'loss:{:.4f}'.format(loss),
    ]
    val_writer.add_scalar('val/acc1', acc1, epoch)
    val_writer.add_scalar('val/acc5', acc5, epoch)
    val_writer.add_scalar('val/loss', loss, epoch)
    val_writer.flush()

    log_output(' '.join(outputs))


def validate_add_tea(val_loader, net, criterion, epoch, val_writer, log_output, tea_net):
    # switch to evaluate mode
    logger.info('eval epoch {}'.format(epoch))
    tea_net.eval()
    net.eval()

    tea_acc1_sum = 0
    acc1_sum = 0
    acc5_sum = 0
    loss = 0
    valdation_num = 0

    for i, (images, labels) in tqdm(enumerate(val_loader)):
        # compute output
        preds, _, _ = net(images)
        tea_preds, _, _ = tea_net(images)
        labels = labels.cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)

        # measure accuracy and record loss
        acc1, acc5 = utils.accuracy(preds, labels, topk=(1, 5))
        tea_acc1, tea_acc5 = utils.accuracy(tea_preds, labels, topk=(1, 5))
        num = labels.size(0)
        valdation_num += num
        tea_acc1_sum += tea_acc1.item() * num
        acc1_sum += acc1.item() * num
        acc5_sum += acc5.item() * num
        loss += criterion(preds, labels).item()

    loss = loss / len(val_loader)
    tea_acc1 = tea_acc1_sum / valdation_num
    acc1 = acc1_sum / valdation_num
    acc5 = acc5_sum / valdation_num

    outputs = [
        "val e:{}".format(epoch),
        'acc1:{:.4f}'.format(acc1),
        'tea_acc1:{:.4f}'.format(tea_acc1),
        'acc5:{:.4f}'.format(acc5),
        'loss:{:.4f}'.format(loss),
    ]
    val_writer.add_scalar('val/acc1', acc1, epoch)
    val_writer.add_scalar('val/tea_acc1', tea_acc1, epoch)
    val_writer.add_scalar('val/acc5', acc5, epoch)
    val_writer.add_scalar('val/loss', loss, epoch)
    val_writer.flush()

    log_output(' '.join(outputs))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-benchmarks',
                        action='store_true', default=False)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--dataset', '-d', default="TinyImageNet", type=str,
                        help='dataset name')
    parser.add_argument('--resolution', default=224, type=int,
                        help='input size')
    parser.add_argument('--batch_size', '-b', default=128,
                        type=int, help='batch size')
    parser.add_argument('--epochs', '-e', default=200,
                        type=int, help='stop epoch')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--decay', default=1e-4,
                        type=float, help='weight decay')
    parser.add_argument('--arch', '-a', default="resnet18", type=str,
                        help='model type')
    parser.add_argument('--sampler', '-s', default="None", type=str,
                        help='sample method')
    parser.add_argument('--mask', action='store_true', default=False)
    parser.add_argument('--temp', default=4.0, type=float,
                        help='temperature scaling')
    parser.add_argument('--omega', default=0.5, type=float,
                        help='ensembling weight')

    parser.add_argument('--loss_lambda', type=float, default=1.0)
    parser.add_argument('--distribution', type=str, default='rank_zipf', choices=['rank_zipf', 'random_uniform', 'random_gaussian',
                                                                                  'random_pareto', 'constant'])
    parser.add_argument('--dense', action='store_true', default=False)
    parser.add_argument('--rank', type=str, default='hard',
                        choices=['hard', 'soft'])
    parser.add_argument('--friend', action='store_true', default=False)
    parser.add_argument('--power', default=1.0, type=float)
    parser.add_argument('--desc', type=str, default='zipf',
                        help='descript your exp')
    parser.add_argument('--run_num', '-r', default='0',
                        type=str, help='running number')
    parser.add_argument('-c', '--continue-train',
                        action='store_true', default=False)
    parser.add_argument('--tea', type=str)
    parser.add_argument('--tea_epoch', type=str, default='epoch_10')
    args = parser.parse_args()

    stop_epoch = args.epochs
    base_lr = args.lr
    arch = args.arch
    # task_name包含方法及关键参数 distribution rank hard/soft lambda friend ...
    task_name = '{}_desc-{}_lambda-{}_distribution-{}_dense-{}_friend-{}_rank-{}_power-{}_run-{}'.format(
        settings.branch_name, args.desc, args.loss_lambda, args.distribution, args.dense, args.friend, args.rank, args.power, args.run_num)
    task_dataset = args.dataset
    if task_dataset.lower() == 'imagenet':
        task_dataset = 'imagenet'

    task_id = None
    # if continue training and taskid.txt avaiable, get task_id from taskid.txt
    if args.continue_train:
        task_id = utils.get_task_id(task_name)

    task = None
    project_name = 'zipf_prior/CVPR/{}/{}'.format(task_dataset, arch)
    log_dir = settings.get_log_dir(project_name, task_name)
    log_model_dir = settings.get_log_model_dir(project_name, task_name)
    print(log_dir)
    print(log_model_dir)
    ensure_dir(log_dir)
    ensure_dir(os.path.join(settings.base_dir,
                            'train_logs', project_name))
    make_symlink_if_not_exists(log_dir, os.path.join(
        settings.base_dir, 'train_logs', project_name, task_name), overwrite=True)

    if False:
        if task_id:
            try:
                task = Task.get_task(task_id=task_id)
                task.init(project_name=project_name,
                          task_name=task_name)
            except ValueError as e:
                task = None
                print('task id not found')
        if task is None:
            task = Task.init(project_name=project_name,
                             task_name=task_name)
            hyperparams = {
                'desc': args.desc, 'lr': args.lr, 'batch_size': args.batch_size, 'loss_lambda': args.loss_lambda, 'distribution': args.distribution,
                'dense': args.dense, 'friend': args.friend, 'rank': args.rank, 'power': args.power
            }
            task.connect(hyperparams)
            utils.write_task_id(task.task_id, task_name)

    # logger
    worklog = WorklogLogger(os.path.join(log_dir, 'worklog.txt'))
    log_output = log_rate_limited(min_interval=1)(worklog.put_line)

    dataset_name = args.dataset
    train_loader, val_loader = datasets.load_dataset(
        dataset_name, batchsize=args.batch_size, train_sampler_method=args.sampler, resolution=args.resolution)
    train_writer = SummaryWriter(
        os.path.join(log_dir, 'train.events'))
    val_writer = SummaryWriter(os.path.join(log_dir, 'val.events'))

    criterion = nn.CrossEntropyLoss().cuda()

    num_classes = train_loader.dataset.num_classes
    # cifar_models
    if 'CIFAR' in args.arch:
        print('CIFAR MODEL')
        net = cifar_models.load_model(args.arch, num_classes)
    else:

        net = models.__dict__[args.arch](
            pretrained=False, num_classes=num_classes)
        if dataset_name == 'CUB':
            utils.load_imagenet_pretrain(net, args.arch)

    net.cuda()
    net = torch.nn.DataParallel(net)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=args.decay)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=args.epochs)

    start_epoch = 0
    if args.continue_train:
        checkpoint = os.path.join(log_model_dir, 'latest')
        if os.path.exists(checkpoint):
            model_info = utils.load_checkpoint(log_model_dir, 'latest')
            start_epoch = model_info['epoch']
            net = model_info['net']
            optimizer = model_info['optimizer']

    tea_model_path = args.tea + '/' + args.tea_epoch
    tea_model_info = torch.load(open(tea_model_path, 'rb'))
    teacher_model = tea_model_info['net']

    for epoch in range(start_epoch, stop_epoch):

        # train for one epoch
        train(train_loader, net, criterion, optimizer,
              epoch, train_writer, log_output, args.loss_lambda, args.mask, args.distribution, args.dense, args.rank, args.friend, args.power, args.batch_size, teacher_model)

        # # evaluate on validation set
        validate(val_loader, net, criterion, epoch,
                 val_writer, log_output)

        # scheduler.step()
        utils.adjust_learning_rate(
            dataset_name, optimizer, epoch, base_lr, stop_epoch)

        save_info = {
            'net': net,
            'optimizer': optimizer,
            'epoch': epoch,
        }
        utils.save_checkpoint(log_model_dir, save_info, 'latest')
        utils.save_checkpoint(log_model_dir, save_info,
                              'epoch_{}'.format(epoch+1))


if __name__ == '__main__':
    main()
# vim: ts=4 sw=4 sts=4 expandtab
