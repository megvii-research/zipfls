import os
import getpass
from git import Repo

base_dir = os.getcwd()
local_repo = Repo(base_dir)
branch_name = local_repo.active_branch.name


def get_logger():
    import logging
    return logging.getLogger(__name__)


logger = get_logger()


def get_log_dir(task_project_name, task_name):
    username = getpass.getuser()
    cache_dir = os.path.join('train_log')
    project_name = base_dir.split('/')[-1]

    # rtld: root train log dir
    npk_rtld = os.path.join(cache_dir, username,
                            project_name, branch_name, task_project_name, task_name)
    os.makedirs(npk_rtld, exist_ok=True)
    logger.info(
        "using data directory for train_log directory: {}".format(npk_rtld))

    return npk_rtld


'''where to write all the logging information during training
(includes saved models)'''
log_dir = get_log_dir('', '')


def get_log_model_dir(task_project_name, task_name):
    username = getpass.getuser()
    project_name = base_dir.split('/')[-1]

    model_path = os.path.join(
        'checkpoints', username, project_name, branch_name, task_project_name, task_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logger.info(
        "Excellent! Model snapshots can be saved to : {}".format(model_path))
    return model_path


'''where to write model snapshots to'''
# log_model_dir = get_log_model_dir()


def get_learning_rate(epoch, idx):
    lr = 0.1 * (0.1 ** (epoch // 30))
    return lr

# def get_learning_rate(epoch, idx):
#     if epoch < 30:
#         return 0.3
#     if epoch < 60:
#         return 0.03
#     if epoch < 90:
#         return 0.003
#     return 0.0003


image_shape = (224, 224)
minibatch_size = 128
stop_epoch = 200
dump_epoch_step = 10

# vim: ts=4 sw=4 sts=4 expandtab
