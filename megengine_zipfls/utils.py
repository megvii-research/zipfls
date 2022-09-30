import numpy as np


def adjust_learning_rate(dataset, optimizer, epoch, base_lr, max_epoch):
    lr = base_lr
    if dataset.lower() in ['cifar10', 'cifar100', 'tinyimagenet']:
        """decrease the learning rate at 100 and 150 epoch"""
        # max_epoch = 10
        if epoch >= 0.5 * max_epoch:
            lr /= 10
        if epoch >= 0.75 * max_epoch:
            lr /= 10
    elif dataset.lower() in ['imagenet', 'inat19', 'inat21']:
        lr = 0.1 * (0.1 ** (epoch // 30))
    elif dataset.lower() in ['inat19_bbn']:
        """decrease the learning rate at 120 and 160 epoch"""
        if epoch >= 120:
            lr /= 10
        if epoch >= 160:
            lr /= 10
    elif dataset.lower() == 'cub':
        lr = base_lr * (0.1 ** (epoch // 80))
    elif dataset.lower() in ['imagenetlt']:
        # cosine decay
        decay_rate = 0.5 * (1 + np.cos(epoch * np.pi / max_epoch))
        lr = base_lr * decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

