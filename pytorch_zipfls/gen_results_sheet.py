import re
from matplotlib import pyplot
import pandas as pd
from pandas import DataFrame
from tabulate import tabulate
from clearml import Task
import os
import settings
from fs import make_symlink_if_not_exists
from tensorboardX import SummaryWriter
from utils import ensure_dir
import pandas as pd


def get_hyperparams(task, params):
    params_rets = []
    # get hypterparams
    for param in params:
        try:
            params_ret = task.data.hyperparams['Args'][param].value
        except Exception as e:
            print(e)
            params_ret = None
        params_rets.append(params_ret)
    return params_rets


def save_to_markdown(data, save_markdown_path):
    data_markdown = data.to_markdown()
    print('saving markdown to {}...'.format(save_markdown_path))
    with open(save_markdown_path, 'w') as f:
        f.write(data_markdown)
    f.close()


def make_results_sheet(datasets, archs, regex, stt_max_epoch, CVPR, result_dir):
    for dataset in datasets:
        for arch in archs:
            if arch == 'None':
                project_name = 'zipf_prior{}/{}'.format(CVPR, dataset)
                tasks = Task.get_tasks(
                    project_name=project_name, task_name=regex)
            else:
                project_name = 'zipf_prior{}/{}/{}'.format(CVPR, dataset, arch)
                tasks = Task.get_tasks(
                    project_name=project_name, task_name=regex)

            res = []
            from tqdm import tqdm
            for task in tqdm(tasks[:]):
                scalars = task.get_reported_scalars()
                branch = task.data.script.branch
                username = task.data.comment.split()[-1].split('@')[0]
                try:
                    top1_vals = scalars['val']['acc1']
                    top5_vals = scalars['val']['acc5']
                    max_acc1 = -1
                    max_acc1_epoch = -1
                    acc5 = None
                    max_epoch = top1_vals['x'][-1]
                    for epoch, acc1 in zip(top1_vals['x'], top1_vals['y']):
                        if epoch > stt_max_epoch:
                            break
                        if acc1 > max_acc1:
                            max_acc1 = acc1
                            max_acc1_epoch = epoch

                    for epoch, _acc5 in zip(top5_vals['x'], top5_vals['y']):
                        if epoch > stt_max_epoch:
                            break
                        if epoch == max_acc1_epoch:
                            acc5 = _acc5

                    # task_params = (desc-{}_multilayer-{}_alpha-{}_upsample-{}_lambda-{}_distribution-{}_dense-{}_friend-{}_rank-{}_power-{}_lr)
                    hyperparams = ['fp16', 'mask', 'loss_lambda',
                                   'distribution', 'dense', 'friend', 'rank', 'power', 'desc', 'lr', 'deeplayer', 'alpha', 'upsample', ]
                    fp16, mask, loss_lambda, dist, dense, friend, rank, power, desc, lr, deeplayer, alpha, upsample, = get_hyperparams(
                        task, hyperparams)
                    res.append((branch, desc, max_acc1, acc5, max_acc1_epoch, dense, friend, rank, power, loss_lambda, lr, deeplayer, alpha, upsample,
                                task.name, task.status, username,
                                fp16, mask, dist, max_epoch))
                except Exception as e:
                    print(e)

            cols = ['branch', 'desc', 'acc1', 'acc5', 'best_epoch', 'dense', 'friend', 'rank', 'power', 'loss_lambda', 'lr', 'deeplayer', 'alpha', 'upsample',
                    'task_name', 'status', 'username',
                    'fp16', 'mask', 'dist', 'current_epoch']
            sort_key = cols.index('acc1')
            print(sort_key)
            for res_i in res:
                print(res_i[sort_key])
            print(res[0][sort_key])
            res.sort(key=lambda x: -x[sort_key])

            data = DataFrame(res, columns=cols)

            save_csv_path = result_dir + \
                '{}/{}-{}-epochs{}.csv'.format(CVPR,
                                               dataset, arch, stt_max_epoch)
            print('saving csv to {}...'.format(save_csv_path))
            data.to_csv(save_csv_path)
            save_markdown_path = result_dir + \
                '{}/{}-{}-epochs{}-markdown.txt'.format(
                    CVPR, dataset, arch, stt_max_epoch)
            save_to_markdown(data, save_markdown_path)
            # print(tabulate(data, headers=cols))

            # print(tabulate(data, headers=cols, tablefmt='pipe'))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument("--format", choices=['markdown', 'wiki'], required=True, help="")
    parser.add_argument('--datasets', '-d', nargs='+', default=['TinyImageNet'], type=str,
                        help='datsets')
    parser.add_argument('--archs', '-a', nargs='+', default=['CIFAR_ResNet18', 'CIFAR_DenseNet121', 'resnet18', 'densenet121'], type=str,
                        help='models type')
    parser.add_argument('--max_epoch', '-m', type=int,
                        default=200, help='统计的最大epoch数')
    parser.add_argument('--regex', required=True, help='')
    parser.add_argument('--cvpr', action='store_true',
                        default=False, help='if task project include CVPR')
    parser.add_argument('--desc', type=str, default='statistics',
                        help='your result key name')
    args = parser.parse_args()

    stt_max_epoch = args.max_epoch
    datasets = args.datasets
    archs = args.archs
    regex = args.regex

    if args.cvpr:
        CVPR = '/CVPR'
    else:
        CVPR = ''

    result_dir = os.path.join(settings.log_dir, args.desc)
    ensure_dir(result_dir)
    ensure_dir(result_dir + CVPR)
    local_dir = os.path.join(
        settings.base_dir, args.desc)
    make_symlink_if_not_exists(result_dir, local_dir, overwrite=True)

    make_results_sheet(datasets, archs, regex, stt_max_epoch, CVPR, result_dir)

    # 实验结果整理

    group_keys = ['desc', 'branch', 'status',
                  'loss_lambda', 'friend', 'dense', 'rank', 'lr', 'deeplayer', 'alpha', 'upsample', 'dist']
    for dataset in datasets:
        for arch in archs:
            csv_path = settings.log_dir + \
                '/statistics{}/{}-{}-epochs{}.csv'.format(
                    CVPR, dataset, arch, stt_max_epoch)
            datas = pd.read_csv(csv_path)
            legal_group_keys = []
            for group_key in group_keys:
                if group_key in datas.columns:
                    legal_group_keys.append(group_key)

            group_mean = datas.groupby(legal_group_keys).agg({
                'acc1': 'mean', 'acc5': 'mean'})
            group_std = datas.groupby(legal_group_keys).agg(
                {'acc1': 'std', 'acc5': 'std'})
            run_time_count = datas.groupby(legal_group_keys).agg({
                'acc1': 'count'})

            acc1_mean_dict = dict(dict(group_mean)['acc1'])
            acc5_mean_dict = dict(dict(group_mean)['acc5'])
            acc1_std_dict = dict(dict(group_std)['acc1'])
            acc5_std_dict = dict(dict(group_std)['acc5'])
            run_time_dict = dict(dict(run_time_count)['acc1'])

            results = []
            new_cols = legal_group_keys + \
                ['acc1_mean', 'acc1_std', 'acc5_mean', 'acc5_std', 'run_time']
            for df_key in acc1_mean_dict.keys():
                acc1_mean = acc1_mean_dict[df_key]
                acc5_mean = acc5_mean_dict[df_key]
                acc1_std = acc1_std_dict[df_key]
                acc5_std = acc5_std_dict[df_key]
                run_time = run_time_dict[df_key]
                result = df_key + (acc1_mean, acc1_std,
                                   acc5_mean, acc5_std, run_time)
                results.append(result)
            sort_key = new_cols.index('acc1_mean')
            results.sort(key=lambda x: -x[sort_key])
            df_res = DataFrame(results, columns=new_cols)
            save_markdown_path = result_dir + \
                '{}/{}-{}-epochs{}-mean_std_markdown.txt'.format(
                    CVPR, dataset, arch, stt_max_epoch)
            save_to_markdown(df_res, save_markdown_path)


if __name__ == '__main__':
    main()
