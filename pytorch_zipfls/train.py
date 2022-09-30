#!/usr/bin/env python3
import utils
import settings
import datasets
from utils import WorklogLogger, log_rate_limited, progress_bar
from fs import make_symlink_if_not_exists
from tensorboardX import SummaryWriter
from utils import ensure_dir
import torchvision.models as models
from tqdm import tqdm
import torch.nn as nn
import time
import torch
import argparse
import os
import cifar_models
from losses import LabelSmoothLoss, bake, hard_friend_rank, soft_friend_rank, zipf_loss, LabelSmoothLoss, gen_pdf, LabelZipfSmoothLoss
from torch import Tensor
import torch.nn.functional as F
from functools import partial, partialmethod
import datetime
import json
import hashlib
from copy import deepcopy

logger = settings.logger


def densenet_train_forward(self, x: Tensor) -> Tensor:
    features = self.features(x)
    out = F.relu(features, inplace=True)
    featmap = out
    n, c, h, w = featmap.size()
    dense_logits_final = self.linear(featmap.permute(0, 2, 3, 1).reshape(
        n*h*w, c)).reshape(n, h*w, -1).argmax(axis=2)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    feat_final = out
    out = self.classifier(out)
    logit_final = out
    return [(logit_final, feat_final, dense_logits_final)]


def densenet_test_forward(self, x: Tensor) -> Tensor:
    features = self.features(x)
    out = F.relu(features, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    feat_final = out
    out = self.classifier(out)
    logit_final = out
    return [(logit_final, feat_final, None)]


def mobilenet_train_forward(self, x: Tensor) -> Tensor:
    x = self.features(x)
    featmap = x
    n, c, h, w = featmap.size()
    dense_logits_final = self.linear(featmap.permute(0, 2, 3, 1).reshape(
        n*h*w, c)).reshape(n, h*w, -1).argmax(axis=2)

    x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
    feat_final = x
    x = self.classifier(x)
    logit_final = x
    return [(logit_final, feat_final, dense_logits_final)]


def mobilenet_test_forward(self, x: Tensor) -> Tensor:
    x = self.features(x)
    x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
    feat_final = x
    x = self.classifier(x)
    logit_final = x
    return [(logit_final, feat_final, None)]


def forward_with_feat_dense(self, x: Tensor, upsample=None) -> Tensor:
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    out2 = x
    x = self.layer3(x)
    out3 = x
    # print(out3.shape)
    x = self.layer4(x)

    def get_dense_info(featmap, linear, gap):
        if upsample:
            tmp = F.upsample(featmap, scale_factor=2., mode=upsample)
        else:
            tmp = featmap
        n, c, h, w = tmp.size()
        dense_logits_argmax = linear(tmp.permute(0, 2, 3, 1).reshape(
            n*h*w, c)).reshape(n, h*w, -1).argmax(axis=2)

        feat = gap(featmap)
        feat = feat.view(featmap.size(0), -1)

        naive_logit = linear(feat)

        return naive_logit, feat, dense_logits_argmax

    logit_2, feat_2, dense_logits_argmax_2 = get_dense_info(
        out2, self.linear2, self.gap2)
    logit_3, feat_3, dense_logits_argmax_3 = get_dense_info(
        out3, self.linear3, self.gap3)

    del out2
    del out3

    logit_final, feat_final, dense_logits_final = get_dense_info(
        x, self.fc, self.avgpool)

    return [(logit_final, feat_final, dense_logits_final), (logit_3, feat_3, dense_logits_argmax_3), (logit_2, feat_2, dense_logits_argmax_2)]


def test_forward(self, x: Tensor) -> Tensor:
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    feat = x
    x = self.fc(x)

    return [(x, feat, None)]


def train(train_loader, args, net, criterion, optimizer, epoch, train_writer, log_output, loss_lambda, loss_mask, distribution,
          dense, rank, friend, power, batchsize, alpha=0.3, upsample=None, deeplayer=2, left_bound=None, right_bound=None, loss_type=None):
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
        outs = net(images)[:deeplayer]
        assert deeplayer >= 1

        outs = outs
        preds = outs[0][0]
        features = outs[0][1]

        dense_logits_list = []
        for layer_out in outs:
            dense_logits_list.append(layer_out[2])
        dense_logits = torch.cat(dense_logits_list, axis=1)

#        n, hw, c = dense_logits.shape
#        dense_preds = dense_logits.reshape(-1, c)
#        dense_labels = torch.broadcast_to(labels[:, None], (n,hw)).flatten()

#        assert deeplayer == 1
#        assert dense == False

        # ****************************************************
        if loss_lambda == 0.0:
            kd_loss = torch.FloatTensor([0.]).to(preds)
        elif dense:
            kd_loss = zipf_loss(args, preds, dense_logits, labels, loss_mask,
                                distribution, dense, rank,  friend, power, left_bound, right_bound)
        else:
            kd_loss = zipf_loss(args, preds, features, labels, loss_mask,
                                distribution, dense, rank,  friend, power, left_bound, right_bound)

        # ****************************************************
        assert len(outs) >= 1
        xent_alphas = [1] + [alpha]*(len(outs) - 1)
        xent_losses = []
        for layer_out in outs:
            _preds = layer_out[0]
            if loss_type == "label_smooth_zipf":
                if dense:
                    xent_loss = criterion(_preds, dense_logits, labels, loss_mask,
                                          distribution, dense, rank,  friend, power, left_bound, right_bound)
                else:
                    xent_loss = criterion(_preds, features, labels, loss_mask,
                                          distribution, dense, rank,  friend, power, left_bound, right_bound)
            else:
                xent_loss = criterion(_preds, labels)
            xent_losses.append(xent_loss)

        total_loss = loss_lambda * kd_loss
        for _alpha, xent_loss in zip(xent_alphas, xent_losses):
            total_loss += _alpha * xent_loss
        # total_loss = loss_1 + alpha*loss_2 + loss_lambda * kd_loss

        # measure accuracy and record loss
        acc1, acc5 = utils.accuracy(preds, labels, topk=(1, 5))

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
            'acc5:{:.4f}'.format(acc5.item()),
            'kd_loss:{:.4f}'.format(kd_loss.item()),
            'total_loss:{:.4f}'.format(total_loss.item()),
        ] + [
            'xent_loss{}:{:.4f}'.format(i, loss.item()) for i, loss in enumerate(xent_losses)
        ]

        if tdata / ttrain > .05:
            outputs += [
                "dp/tot: {:.2g}".format(tdata / ttrain),
            ]

        step = epoch * minibatch_count + i
        if step % 200 == 0:
            train_writer.add_scalar('train/acc1', acc1.item(), step)
            train_writer.add_scalar('train/acc5', acc5.item(), step)
            for i, loss in enumerate(xent_losses):
                train_writer.add_scalar(
                    'train/xent_loss{}'.format(i), loss.item(), step)
            train_writer.add_scalar('train/kd_loss', kd_loss.item(), step)
            train_writer.add_scalar(
                'train/total_loss', total_loss.item(), step)
            train_writer.add_scalar('train/lr', learning_rate, step)
            train_writer.flush()
        log_output(' '.join(outputs))

        # progress_bar(i, minibatch_count, 'Train: Epoch: {} | CE loss: {:.2f} | Zipfs loss: {:.2f} | Acc1: {:.2f} | Acc5: {:.2f}'.format(
        #     epoch, xent_losses[0], kd_loss.item(), acc1.item(), acc5.item()))


def validate(val_loader, net, criterion, epoch, val_writer, log_output, best_acc1):
    # switch to evaluate mode
    # minibatch_count = len(val_loader)
    logger.info('eval epoch {}'.format(epoch))
    net.eval()

    acc1_sum = 0
    acc5_sum = 0
    loss = 0
    valdation_num = 0

    for i, (images, labels) in tqdm(enumerate(val_loader)):
        # compute output
        preds = net(images)[0][0]
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

    if acc1 > best_acc1:
        best_acc1 = acc1

    outputs = [
        "val e:{}".format(epoch),
        'acc1:{:.4f}'.format(acc1),
        'best_acc1:{:.4f}'.format(best_acc1),
        'acc5:{:.4f}'.format(acc5),
        'loss:{:.4f}'.format(loss),
    ]
    val_writer.add_scalar('val/acc1', acc1, epoch)
    val_writer.add_scalar('val/acc5', acc5, epoch)
    val_writer.add_scalar('val/loss', loss, epoch)
    val_writer.flush()

    log_output(' '.join(outputs))

    # progress_bar(i, minibatch_count, 'Val: Epoch: {} | CE loss: {:.3f} | Acc1: {:.3f} | Acc5: {:.3f}'.format(
    #     epoch, loss, acc1, acc5))

    return acc1, best_acc1, acc5, loss


def hash_opt(args):
    print(args)
    args = deepcopy(args)
    args.run_num = -1
    key = hashlib.sha256(json.dumps(
        vars(args), sort_keys=True).encode('utf-8')).hexdigest()
    return key[:6]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-benchmarks',
                        action='store_true', default=False)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--dataset', '-d', default="TinyImageNet", type=str,
                        help='dataset name')
    parser.add_argument('--data_dir', default='./datas', type=str)
    parser.add_argument('--batch_size', '-b', default=128,
                        type=int, help='batch size')
    parser.add_argument('--val_batch_size', default=None,
                        type=int, help='validation batch size')
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

    # label-smooth
    parser.add_argument('--criterion', default='ce', type=str,
                        choices=['ce', 'label_smooth', 'label_smooth_zipf'])
    parser.add_argument('--smooth_ratio', default=0.2, type=float)

    # zipf-law
    parser.add_argument('--loss_lambda', type=float, default=1.0)
    parser.add_argument('--distribution', type=str, default='rank_zipf', choices=['rank_zipf', 'random_uniform', 'random_gaussian',
                                                                                      'random_pareto', 'constant', 'unbiased_zipf', 'sorted_pareto', 'linear_decay', 'exponential_decay'])
    parser.add_argument('--left_bound', type=float, default=0.001)
    parser.add_argument('--right_bound', type=float, default=1.0)
    parser.add_argument('--dense', action='store_true', default=False)
    parser.add_argument('--rank', type=str, default='hard',
                        choices=['hard', 'soft'])
    parser.add_argument('--friend', action='store_true', default=False)
    parser.add_argument('--power', default=1.0, type=float)
    # multi layer params
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--upsample', default=None, type=str,
                        choices=['bilinear', 'bicubic', 'None'])
    parser.add_argument('--deeplayer', default=1, type=int, choices=[1, 2, 3])
    # end multi layer
    parser.add_argument('--desc', type=str, default='zipf',
                        help='descript your exp')
    # decay params
    parser.add_argument("--smallest_prob", default=None,
                        type=float, help="")  # linear_decay params
    parser.add_argument("--exp_lambda", default=None,
                        type=float, help="")  # exponential_decay params
    parser.add_argument('--sgpu', default=0, type=int,
                        help='gpu start idx')
    parser.add_argument('--ngpus', default=1, type=int, help='number of gpu')
    parser.add_argument('--run_num', '-r', default='0',
                        type=str, help='running number')
    parser.add_argument('-c', '--continue-train',
                        action='store_true', default=False)
    args = parser.parse_args()

    if args.upsample == 'None':
        args.upsample = None

    stop_epoch = args.epochs
    base_lr = args.lr
    arch = args.arch
    # task_name包含方法及关键参数 distribution rank hard/soft lambda friend ...
    task_name = '{}_desc-{}_multilayer-{}_alpha-{}_upsample-{}_lambda-{}_distribution-{}_dense-{}_fri-{}_rank-{}_lb-{}_rb-{}_power-{}_hash-{}_run-{}'.format(
        settings.branch_name, args.desc, args.deeplayer, args.alpha, args.upsample, args.loss_lambda, args.distribution, args.dense, args.friend, args.rank, args.left_bound, args.right_bound, args.power, hash_opt(args), args.run_num)
    task_dataset = args.dataset

    project_name = 'zipf_prior/CVPR/{}/{}'.format(task_dataset, arch)
    log_dir = settings.get_log_dir(project_name, task_name)
    log_model_dir = settings.get_log_model_dir(project_name, task_name)
    print(log_dir)
    print(log_model_dir)
    ensure_dir(log_dir)

    # logger
    worklog = WorklogLogger(os.path.join(log_dir, 'worklog.txt'))
    log_output = log_rate_limited(min_interval=1)(worklog.put_line)
    log_output(json.dumps(vars(args), sort_keys=True).encode('utf-8'))

    dataset_name = args.dataset
    train_loader, val_loader = datasets.load_dataset(
        dataset_name, args.data_dir, batchsize=args.batch_size, train_sampler_method=args.sampler,
        val_batchsize=args.val_batch_size)
    train_writer = SummaryWriter(
        os.path.join(log_dir, 'train.events'))
    val_writer = SummaryWriter(os.path.join(log_dir, 'val.events'))

    if args.criterion == 'ce':
        criterion = nn.CrossEntropyLoss().cuda()
        criterion_val = criterion
    elif args.criterion == 'label_smooth':
        criterion = LabelSmoothLoss(args.smooth_ratio).cuda()
        criterion_val = criterion
    elif args.criterion == "label_smooth_zipf":
        criterion = LabelZipfSmoothLoss(args.smooth_ratio).cuda()
        criterion_val = nn.CrossEntropyLoss().cuda()

    num_classes = train_loader.dataset.num_classes
    # cifar_models
    if 'CIFAR' in args.arch:
        print('CIFAR MODEL')
        net = cifar_models.load_model(
            args.arch, num_classes, upsample=args.upsample)
    else:
        net = models.__dict__[args.arch](
            pretrained=False, num_classes=num_classes)
        if dataset_name == 'CUB':
            utils.load_imagenet_pretrain(net, args.arch)

    if args.dataset == 'ImageNet' or args.dataset == 'INAT21':
        if arch == 'resnet18':
            setattr(net, 'linear2', nn.Linear(128, num_classes))
            setattr(net, 'linear3', nn.Linear(256, num_classes))

            setattr(net, 'gap2', nn.AvgPool2d(28))
            setattr(net, 'gap3', nn.AvgPool2d(14))

        elif arch in ['resnet50', 'resnet101', 'resnext50_32x4d', 'resnext101_32x8d']:
            setattr(net, 'linear2', nn.Linear(512, num_classes))
            setattr(net, 'linear3', nn.Linear(1024, num_classes))

            setattr(net, 'gap2', nn.AvgPool2d(28))
            setattr(net, 'gap3', nn.AvgPool2d(14))
        elif arch == 'mobilenet_v2':
            setattr(net, 'linear', nn.Linear(1280, num_classes))
        elif arch == 'densenet121':
            setattr(net, 'linear', nn.Linear(1024, num_classes))

    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=list(
        range(args.sgpu, args.sgpu + args.ngpus)))

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

    best_acc1 = 0.0
    for epoch in range(start_epoch, stop_epoch):

        # train for one epoch
        setattr(models.__dict__['ResNet'], 'forward', partialmethod(
            forward_with_feat_dense, upsample=args.upsample))
        setattr(models.__dict__['DenseNet'], 'forward', densenet_train_forward)
        setattr(models.__dict__['MobileNetV2'],
                'forward', mobilenet_train_forward)
        train(train_loader, args, net, criterion, optimizer,
              epoch, train_writer, log_output, args.loss_lambda, args.mask, args.distribution, args.dense, args.rank,
              args.friend, args.power, args.batch_size, args.alpha, args.upsample, args.deeplayer, args.left_bound, args.right_bound, args.criterion)

        # # evaluate on validation set
        setattr(models.__dict__['ResNet'], 'forward', test_forward)
        setattr(models.__dict__['DenseNet'], 'forward', densenet_test_forward)
        setattr(models.__dict__['MobileNetV2'],
                'forward', mobilenet_test_forward)
        acc1, best_acc1, _, _ = validate(val_loader, net, criterion_val,
                                         epoch, val_writer, log_output, best_acc1)

        # scheduler.step()
        utils.adjust_learning_rate(
            dataset_name, optimizer, epoch, base_lr, stop_epoch)

        save_info = {
            'net': net,
            'optimizer': optimizer,
            'epoch': epoch,
        }
        if epoch % 50 == 0:
            utils.save_checkpoint(log_model_dir, save_info, 'latest')
            utils.save_checkpoint(log_model_dir, save_info,
                                  'epoch_{}'.format(epoch+1))


if __name__ == '__main__':
    main()
# vim: ts=4 sw=4 sts=4 expandtab
