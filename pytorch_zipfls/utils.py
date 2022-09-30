#!/usr/bin/env python3
import os
from pathlib import Path
import time
import torch
import getpass
import settings
import numpy as np
import torchvision.models as models
import logging
import sys
import functools
import shutil


class LogFormatter(logging.Formatter):
    log_fout = None
    date_full = '[%(asctime)s %(lineno)d@%(filename)s:%(name)s] '
    date = '%(asctime)s '
    msg = '%(message)s'
    max_lines = 256

    def _color_exc(self, msg):
        return '\x1b[34m{}\x1b[0m'.format(msg)

    def _color_dbg(self, msg):
        return '\x1b[36m{}\x1b[0m'.format(msg)

    def _color_warn(self, msg):
        return '\x1b[1;31m{}\x1b[0m'.format(msg)

    def _color_err(self, msg):
        return '\x1b[1;4;31m{}\x1b[0m'.format(msg)

    def _color_omitted(self, msg):
        return '\x1b[35m{}\x1b[0m'.format(msg)

    def _color_normal(self, msg):
        return msg

    def _color_date(self, msg):
        return '\x1b[32m{}\x1b[0m'.format(msg)

    def format(self, record):
        if record.levelno == logging.DEBUG:
            mcl, mtxt = self._color_dbg, 'DBG'
        elif record.levelno == logging.WARNING:
            mcl, mtxt = self._color_warn, 'WRN'
        elif record.levelno == logging.ERROR:
            mcl, mtxt = self._color_err, 'ERR'
        else:
            mcl, mtxt = self._color_normal, ''

        if mtxt:
            mtxt += ' '

        if self.log_fout:
            self.__set_fmt(self.date_full + mtxt + self.msg)
            formatted = super(LogFormatter, self).format(record)
            nr_line = formatted.count('\n') + 1
            if nr_line >= self.max_lines:
                head, body = formatted.split('\n', 1)
                formatted = '\n'.join([
                    head,
                    'BEGIN_LONG_LOG_{}_LINES{{'.format(nr_line - 1),
                    body,
                    '}}END_LONG_LOG_{}_LINES'.format(nr_line - 1)
                ])
            self.log_fout.write(formatted)
            self.log_fout.write('\n')
            self.log_fout.flush()

        self.__set_fmt(self._color_date(self.date) + mcl(mtxt + self.msg))
        formatted = super(LogFormatter, self).format(record)

        if record.exc_text or record.exc_info:
            # handle exception format
            b = formatted.find('Traceback ')
            if b != -1:
                s = formatted[b:]
                s = self._color_exc('  ' + s.replace('\n', '\n  '))
                formatted = formatted[:b] + s

        nr_line = formatted.count('\n') + 1
        if nr_line >= self.max_lines:
            lines = formatted.split('\n')
            remain = self.max_lines//2
            removed = len(lines) - remain * 2
            if removed > 0:
                mid_msg = self._color_omitted(
                    '[{} log lines omitted (would be written to output file '
                    'if set_output_file() has been called;\n'
                    ' the threshold can be set at '
                    'LogFormatter.max_lines)]'.format(removed))
                formatted = '\n'.join(
                    lines[:remain] + [mid_msg] + lines[-remain:])

        return formatted

    if sys.version_info.major < 3:
        def __set_fmt(self, fmt):
            self._fmt = fmt
    else:
        def __set_fmt(self, fmt):
            self._style._fmt = fmt


def get_logger(name=None, formatter=LogFormatter):

    logger = logging.getLogger(name)
    if getattr(logger, '_init_done__', None):
        return logger
    logger._init_done__ = True
    logger.propagate = False
    _default_level = 1
    logger.setLevel(_default_level)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter(datefmt='%d %H:%M:%S'))
    handler.setLevel(0)
    del logger.handlers[:]
    logger.addHandler(handler)
    return logger


class WorklogFormatter(LogFormatter):
    log_fout = None


class WorklogLogger:
    def __init__(self, log_file):
        WorklogFormatter.log_fout = open(log_file, 'a')
        self.logger = get_logger(__name__, formatter=WorklogFormatter)

    def put_line(self, line):
        self.logger.info(line)

    def put_tensor_rms(self, tensor_name, tensor_value, step):
        rms = np.sqrt((tensor_value ** 2).mean())
        self.put_line('{}: {}'.format(tensor_name, rms))


def log_rate_limited(min_interval=1):

    def decorator(should_record):
        last = 0

        @functools.wraps(should_record)
        def wrapper(*args, **kwargs):
            nonlocal last
            if time.time() - last < min_interval:
                return False
            ret = should_record(*args, **kwargs)
            last = time.time()
            return ret
        return wrapper

    return decorator


def ensure_dir(path: Path):
    """create directories if *path* does not exist"""

    path = Path(path)
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(log_model_dir, save_info, name):
    ckp_path = os.path.join(log_model_dir, name)
    while True:
        try:
            with open(ckp_path, 'wb') as fobj:
                torch.save(save_info, fobj)
            break
        except Exception as e:
            print(e)


def load_checkpoint(log_model_dir, name):
    ckp_path = os.path.join(log_model_dir, name)
    return torch.load(open(ckp_path, 'rb'))


def load_imagenet_pretrain(net, model_name):
    pretrain_dict = models.__dict__[model_name](pretrained=True).state_dict()
    pretrain_dict = {k: v if not 'fc' in k else net.state_dict()[
        k] for k, v in pretrain_dict.items()}
    net.load_state_dict(pretrain_dict)


def adjust_learning_rate(dataset, optimizer, epoch, base_lr, max_epoch):
    lr = base_lr
    if dataset.lower() in ['cifar10', 'cifar100', 'tinyimagenet']:
        """decrease the learning rate at 100 and 150 epoch"""
        if epoch >= 0.5 * max_epoch:
            lr /= 10
        if epoch >= 0.75 * max_epoch:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif dataset.lower() in ['imagenet', 'inat19', 'inat21']:
        lr = 0.1 * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif dataset.lower() in ['inat19_bbn']:
        """decrease the learning rate at 120 and 160 epoch"""
        if epoch >= 120:
            lr /= 10
        if epoch >= 160:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif dataset.lower() == 'cub':
        lr = base_lr * (0.1 ** (epoch // 80))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif dataset.lower() in ['imagenetlt']:
        # cosine decay
        decay_rate = 0.5 * (1 + np.cos(epoch * np.pi / max_epoch))
        lr = base_lr * decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def get_task_id(task_name):
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    username = getpass.getuser()
    task_file = os.path.join(
        BASE_DIR, 'train_logs', '{}_{}_taskid.txt'.format(username, task_name))
    print(task_file)
    task_id = None
    if os.path.isfile(task_file):
        with open(task_file, 'r') as fin:
            lines = fin.readlines()
            if len(lines) > 0:
                task_id = lines[0].strip()

    return task_id


def write_task_id(task_id, task_name):
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    username = getpass.getuser()
    task_file = os.path.join(
        BASE_DIR, 'train_logs', '{}_{}_taskid.txt'.format(username, task_name))
    with open(task_file, 'w') as fout:
        fout.write(task_id+'\n')


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


_, term_width = shutil.get_terminal_size()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 116.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    # print('msg', msg)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


if __name__ == '__main__':
    pass

# vim: ts=4 sw=4 sts=4 expandtab
