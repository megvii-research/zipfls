#!/usr/bin/env python3
import utils
import settings
import datasets
from from utils import WorklogLogger, log_rate_limited
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
# import timm
from clearml import Task
import cifar_models
from losses import LabelSmoothLoss, bake, hard_friend_rank, soft_friend_rank, zipf_loss, LabelSmoothLoss, gen_pdf, LabelZipfSmoothLoss, KDLoss, knowledge_ensemble, CS_KDLoss
from torch import Tensor
import torch.nn.functional as F
from functools import partial, partialmethod
import datetime
import json
import hashlib
from copy import deepcopy
from method_configs import method_params
logger = settings.logger


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


def resnet_forward(self, x: Tensor) -> Tensor:
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

    return x, feat


def cifar_forward(self, x: Tensor) -> Tensor:
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.gap(x)
    feat = x.view(x.size(0), -1)
    x = self.linear(feat)

    return x, feat


def densenet_forward(self, x):
    out = self.conv1(x)
    out = self.trans1(self.dense1(out))
    out = self.trans2(self.dense2(out))
    out = self.trans3(self.dense3(out))
    out = self.dense4(out)
    out = F.avg_pool2d(F.relu(self.bn(out)), 4)
    out = out.view(out.size(0), -1)
    prob = self.linear(out)
    return prob, out


def train(kd_method, hyperparams, train_loader, net, criterion, optimizer, epoch, train_writer, log_output, loss_lambda, loss_mask, distribution,
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

        # ****************************************************
        if kd_method == 'cs-kd':
            # attention: only works while using friend sampler
            T = hyperparams['temp_factor']
            preds_half, _ = net(images[::2])
            with torch.no_grad():
                preds_friend, _ = net(images[1::2])
            min_lens = min(preds_half.size(0), preds_friend.size(0))
            preds_half = preds_half[:min_lens]
            preds_friend = preds_friend[:min_lens]

            kdloss = CS_KDLoss(temp_factor=4.0)
            kd_loss = kdloss(preds_half, preds_friend.detach())

            labels = labels[::2]
            labels = labels[:min_lens]

        elif kd_method == 'tf-kd':
            preds, features = net(images)
            T = hyperparams['reg_temperature']
            multiplier = hyperparams['multiplier']
            correct_prob = 0.99    # the probability for correct class in u(k)
            K = preds.size(1)
            teacher_soft = torch.ones_like(preds).cuda()
            teacher_soft = teacher_soft*(1-correct_prob)/(K-1)  # p^d(k)
            for i in range(preds.shape[0]):
                teacher_soft[i, labels[i]] = correct_prob
            kd_loss = nn.KLDivLoss()(F.log_softmax(preds, dim=1),
                                     F.softmax(teacher_soft/T, dim=1))*multiplier
        elif kd_method == 'bake':
            preds, features = net(images)
            T = hyperparams['temp_factor']
            omega = hyperparams['omega']
            kd_loss = bake(features.detach(), preds.detach(),
                           temp=T, omega=omega)

        # xent loss
        if kd_method == 'cs-kd':
            xent_loss = criterion(preds_half, labels)
            acc1, acc5 = utils.accuracy(preds_half, labels, topk=(1, 5))
            # preds, _ = net(images)
            # xent_loss = criterion(preds, labels)
            # acc1, acc5 = utils.accuracy(preds, labels, topk=(1, 5))
        else:
            xent_loss = criterion(preds, labels)
            acc1, acc5 = utils.accuracy(preds, labels, topk=(1, 5))

        total_loss = xent_loss + loss_lambda * kd_loss

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
            'xent_loss:{:.4f}'.format(xent_loss.item()),
            'total_loss:{:.4f}'.format(total_loss.item()),
        ]
        if tdata / ttrain > .05:
            outputs += [
                "dp/tot: {:.2g}".format(tdata / ttrain),
            ]

        step = epoch * minibatch_count + i
        if step % 200 == 0:
            train_writer.add_scalar('train/acc1', acc1.item(), step)
            train_writer.add_scalar('train/acc5', acc5.item(), step)
            train_writer.add_scalar('train/kd_loss', kd_loss.item(), step)
            train_writer.add_scalar('train/xent_loss', xent_loss.item(), step)
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
        preds, _ = net(images)
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
    parser.add_argument('--resolution', default=224, type=int,
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
    parser.add_argument('--arch', '-a', default="resnet18", type=str,
                        help='model type')
    parser.add_argument('--sampler', '-s', default="None", type=str,
                        help='sample method')
    parser.add_argument('--friend_num', default=1, type=int)
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
    parser.add_argument('--distribution', type=str, default='unbiased_zipf', choices=['rank_zipf', 'random_uniform', 'random_gaussian',
                                                                                      'random_pareto', 'constant', 'unbiased_zipf', 'sorted_pareto'])
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
    parser.add_argument('--run_num', '-r', default='0',
                        type=str, help='running number')
    parser.add_argument('-c', '--continue-train',
                        action='store_true', default=False)
    parser.add_argument('--kd_method', type=str, default='rank',
                        choices=['rank', 'cs-kd', 'tf-kd', 'bake'])
    args = parser.parse_args()

    if args.upsample == 'None':
        args.upsample = None

    stop_epoch = args.epochs
    base_lr = args.lr
    arch = args.arch
    kd_method = args.kd_method
    # task_name包含方法及关键参数 distribution rank hard/soft lambda friend ...
    task_name = 'method-{}_branch-{}_desc-{}_multilayer-{}_alpha-{}_upsample-{}_lambda-{}_distribution-{}_dense-{}_rank-{}_lb-{}_rb-{}_power-{}_hash-{}_run-{}'.format(
        kd_method, settings.branch_name, args.desc, args.deeplayer, args.alpha, args.upsample, args.loss_lambda, args.distribution, args.dense, args.rank, args.left_bound, args.right_bound, args.power, hash_opt(args), args.run_num)
    full_task_name = 'method-{}_branch-{}_desc-{}_multilayer-{}_alpha-{}_upsample-{}_lambda-{}_distribution-{}_dense-{}_friend-{}_rank-{}_left_bound-{}_right_bound-{}_power-{}_lr-{}-sr-{}_bsz-{}_cri-{}_time-{}_run-{}'.format(
        kd_method, settings.branch_name, args.desc, args.deeplayer, args.alpha, args.upsample, args.loss_lambda, args.distribution, args.dense, args.friend, args.rank, args.left_bound, args.right_bound, args.power, args.lr, args.smooth_ratio, args.batch_size, args.criterion, datetime.datetime.now().time(), args.run_num)
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
                'dense': args.dense, 'friend': args.friend, 'rank': args.rank, 'power': args.power, 'lr': args.lr,
            }
            task.connect(hyperparams)
            utils.write_task_id(task.task_id, task_name)

    # logger
    worklog = WorklogLogger(os.path.join(log_dir, 'worklog.txt'))
    log_output = log_rate_limited(min_interval=1)(worklog.put_line)
    log_output(full_task_name)

    dataset_name = args.dataset
    train_loader, val_loader = datasets.load_dataset(
        dataset_name, batchsize=args.batch_size, train_sampler_method=args.sampler, resolution=args.resolution,
        val_batchsize=args.val_batch_size, friend_num=args.friend_num)
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
        setattr(net, 'linear2', nn.Linear(512, num_classes))
        setattr(net, 'linear3', nn.Linear(1024, num_classes))

        setattr(net, 'gap2', nn.AvgPool2d(28))
        setattr(net, 'gap3', nn.AvgPool2d(14))

    net.cuda()
    net = torch.nn.DataParallel(net)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=args.decay)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=args.epochs)

    start_epoch = 0
    if args.continue_train:
        checkpoint = os.path.join(log_model_dir, 'latest')
        if os.path.exist(checkpoint):
            model_info = utils.load_checkpoint(log_model_dir, 'latest')
            start_epoch = model_info['epoch']
            net = model_info['net']
            optimizer = model_info['optimizer']

    hyperparams = method_params[kd_method][arch]

    for epoch in range(start_epoch, stop_epoch):

        # train for one epoch
        setattr(models.__dict__['ResNet'], 'forward', resnet_forward)
        setattr(cifar_models.__dict__[
                'CIFAR_ResNet'], 'forward', cifar_forward)
        setattr(cifar_models.__dict__[
                'CIFAR_DenseNet'], 'forward', densenet_forward)
        train(kd_method, hyperparams, train_loader, net, criterion, optimizer,
              epoch, train_writer, log_output, args.loss_lambda, args.mask, args.distribution, args.dense, args.rank,
              args.friend, args.power, args.batch_size, args.alpha, args.upsample, args.deeplayer, args.left_bound, args.right_bound, args.criterion)

        # # evaluate on validation set
        setattr(models.__dict__['ResNet'], 'forward', resnet_forward)
        setattr(cifar_models.__dict__[
                'CIFAR_ResNet'], 'forward', cifar_forward)
        setattr(cifar_models.__dict__[
                'CIFAR_DenseNet'], 'forward', densenet_forward)
        validate(val_loader, net, criterion_val, epoch, val_writer, log_output)

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
