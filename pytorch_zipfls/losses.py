import torch.nn as nn
import torch.nn.functional as F
import torch
# from torchsort import soft_rank, soft_sort
import numpy as np
from scipy.stats import rankdata
from torch import Tensor
from zipf_utils import softmax, gen_non_target_class_distribution, kl_loss, gen_zipf_law_distribution_instance_wise, \
    gen_pdf_from_friend_soft, gen_pdf_from_friend_hard, gen_pdf_from_dense_argmax, gen_pdf_unbiased, \
    gen_pdf_exponential_decay, gen_pdf_linear_decay
import time

class CS_KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(CS_KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss


def ls_zipf_loss(alpha, logits, feats, labels, loss_mask=False, distribution='rank_zipf', dense=False, rank='hard', friend=False, power=1.0, left_bound=0.01, right_bound=1.0):
    num_classes = logits.size(1)
    target_zipf = gen_pdf(logits, feats, labels, loss_mask, distribution,
                          dense, rank, friend, power, left_bound, right_bound, with_target=True)
    target_zipf = torch.Tensor(target_zipf).cuda()
    target_one_hot = torch.nn.functional.one_hot(
        labels, num_classes=num_classes)
    target_smooth = (1 - alpha) * target_one_hot + alpha*target_zipf
    output_log_softmax = (-1)*torch.log_softmax(logits, dim=1)
    loss = torch.mul(output_log_softmax, target_smooth).sum(dim=1)
    loss = loss.mean()
    return loss


class LabelZipfSmoothLoss(nn.Module):
    def __init__(self, alpha):
        super(LabelZipfSmoothLoss, self).__init__()
        self.alpha = alpha
        assert 0.0 <= alpha < 1.0

    def forward(self, logits, feats, labels, loss_mask, distribution, dense, rank,  friend, power, left_bound, right_bound):
        return ls_zipf_loss(self.alpha, logits, feats, labels, loss_mask, distribution, dense, rank,  friend, power, left_bound, right_bound)


def label_smooth_loss(output, target, alpha):
    num_classes = output.size(1)
    target_one_hot = torch.nn.functional.one_hot(
        target, num_classes=num_classes)
    target_smooth = (1 - alpha) * target_one_hot + alpha / \
        (num_classes-1) * (1 - target_one_hot)
    output_log_softmax = (-1)*torch.log_softmax(output, dim=1)
    loss = torch.mul(output_log_softmax, target_smooth).sum(dim=1)
    loss = loss.mean()
    return loss


class LabelSmoothLoss(nn.Module):
    def __init__(self, alpha):
        super(LabelSmoothLoss, self).__init__()
        self.alpha = alpha
        assert 0.0 <= alpha < 1.0

    def forward(self, logits, target):
        return label_smooth_loss(logits, target, self.alpha)


class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, target)*(self.temp_factor**2)/input.size(0)
        return loss


def knowledge_ensemble(feats, logits, temp=4.0, omega=0.5):

    softmax = nn.Softmax(dim=1)
    batch_size = logits.size(0)
    masks = torch.eye(batch_size)
    masks = masks.cuda()
    feats = nn.functional.normalize(feats, p=2, dim=1)
    logits = nn.functional.softmax(logits/temp, dim=1)
    W = torch.matmul(feats, feats.permute(1, 0)) - masks * 1e9
    W = softmax(W)
    W = (1 - omega) * torch.inverse(masks - omega * W)
    return torch.matmul(W, logits)


def bake(features, preds, temp=4.0, omega=0.5):
    with torch.no_grad():
        kd_targets = knowledge_ensemble(
            features.detach(), preds.detach(), temp=temp, omega=omega)
    kdloss = KDLoss(temp)
    kd_loss = kdloss(preds, kd_targets.detach())
    return kd_loss


#################################for testing########################################
def hard_friend_rank(preds, features, labels, loss_mask):
    non_target_pdf = gen_pdf_from_friend_hard(preds, features, labels)
    non_target_pdf = torch.Tensor(non_target_pdf).cuda()
    dist_loss = kl_loss(preds, labels, non_target_pdf, do_mask=loss_mask)
    return dist_loss


def soft_friend_rank(preds, features, labels, loss_mask):
    non_target_pdf = gen_pdf_from_friend_soft(preds, features, labels)
    non_target_pdf = torch.Tensor(non_target_pdf).cuda()
    dist_loss = kl_loss(preds, labels, non_target_pdf, do_mask=loss_mask)
    return dist_loss


def dense_rank(preds, features, labels, loss_mask):
    non_target_pdf = gen_pdf_from_dense_argmax(preds, features, labels)
    non_target_pdf = torch.Tensor(non_target_pdf).cuda()
    dist_loss = kl_loss(preds, labels, non_target_pdf, do_mask=loss_mask)
    return dist_loss
#################################for testing########################################


def kl_loss(output, label, pdf, do_mask=False):

    mask = torch.ones_like(output).scatter_(1, label.unsqueeze(1), 0.)
    res = output[mask.bool()].view((output.shape[0], output.shape[1]-1))
    res = F.log_softmax(res, dim=1)
    loss = F.kl_div(res, pdf, reduction='none').sum(axis=1)

    if do_mask:
        loss_mask = (torch.argmax(output, 1) == label)
        if loss_mask.any():
            loss = loss * loss_mask
            loss = loss.sum() / loss_mask.sum()
        else:
            loss = loss.mean() * 0
    else:
        loss = loss.mean()

    return loss


def zipf_loss(args, logits, feats, labels, loss_mask=False, distribution='rank_zipf', dense=False, rank='hard', friend=False, power=1.0, left_bound=0.01, right_bound=1.0):
    non_target_pdf = gen_pdf(args,
                             logits, feats, labels, loss_mask, distribution, dense, rank, friend, power, left_bound, right_bound)
    non_target_pdf = torch.Tensor(non_target_pdf).cuda()
    dist_loss = kl_loss(logits, labels, non_target_pdf, do_mask=loss_mask)
    return dist_loss


def get_masked_distribution(args, rank, dis_type, power, dense, left_bound, mask, rank_max, right_bound, with_target):
    assert dis_type in ['rank_zipf', 'random_uniform',
                        'random_gaussian', 'random_pareto', 'constant', 'unbiased_zipf', 'sorted_pareto', 'linear_decay', 'exponential_decay']
    n, c = rank.shape
    if dis_type == 'random_uniform':
        dist = np.random.rand(*rank.shape)
    if dis_type == 'random_gaussian':
        dist = np.abs(np.random.randn(*rank.shape))
    if dis_type == 'random_pareto':
        dist = np.random.pareto(1, rank.shape)
    if dis_type == 'sorted_pareto':
        index = np.argsort(rank)
        pareto = np.sort(-np.random.pareto(1, rank.shape))
        dist = pareto[np.arange(n)[:, None], index]
    if dis_type == 'linear_decay':
        assert args.smallest_prob is not None
        assert args.smallest_prob <= 1 / c
        s = args.smallest_prob
        dist = gen_pdf_linear_decay(rank, c, s, rank_max=rank_max)
        #assert np.allclose(dist.sum(axis=1), 1), dist.sum(axis=1)

    if dis_type == 'exponential_decay':
        assert args.exp_lambda is not None
        #dist = gen_pdf_exponential_decay(rank, c, dense=dense, left_bound=left_bound,
        #                                 right_bound=right_bound, exp_lambda=args.exp_lambda, rank_max=rank_max)
        dist = np.exp(-args.exp_lambda * (rank-1))

    if dis_type == 'constant':
        dist = np.ones(rank.shape)
    if dis_type == 'rank_zipf':
        dist = 1 / rank
        dist = dist ** power
    if dis_type == 'unbiased_zipf':
        dist = gen_pdf_unbiased(rank, c, power=power, dense=dense,
                                left_bound=left_bound, right_bound=right_bound, rank_max=rank_max)
    if with_target:
        dist = mask*dist
    else:
        dist = dist[mask.bool()].reshape((n, c-1))
    dist = dist / dist.sum(axis=1, keepdims=True)

    return dist


def get_distribution(rank, dis_type, power, dense, left_bound, rank_max):
    assert dis_type in ['rank_zipf', 'random_uniform',
                        'random_gaussian', 'random_pareto', 'constant']
    if dis_type == 'random_uniform':
        dist = np.random.rand(*rank.shape)
    if dis_type == 'random_gaussian':
        dist = np.abs(np.random.randn(*rank.shape))
    if dis_type == 'random_pareto':
        dist = np.random.pareto(1, rank.shape)
    if dis_type == 'constant':
        dist = np.ones(rank.shape)
    if dis_type == 'rank_zipf':
        dist = 1 / rank
        dist = dist ** power
    n, c = rank.shape
    if dis_type == 'unbiased_zipf':
        dist = gen_pdf_unbiased(
            rank, c, power=power, dense=dense, left_bound=left_bound, rank_max=rank_max)
    dist = dist / dist.sum(axis=1, keepdims=True)

    return dist


def gen_pdf(args, logits, feats, labels, loss_mask=False, distribution='rank_zipf', dense=False, rank='hard', friend=False, power=1.0, left_bound=0.01, right_bound=1.0, with_target=False):
    logits = logits.detach().cpu()
    feats = feats.detach().cpu()
    labels = labels.detach().cpu()

    n = logits.shape[0]
    # feats = feats.detach()
    # logits = logits.detach()
    n, c = logits.shape

    if friend:
        A = (feats ** 2).sum(axis=1, keepdims=True)
        B = A.T
        dot = torch.matmul(feats, feats.T)
        square_distance = A + B - 2*dot
        square_distance[torch.arange(n), torch.arange(n)] = 1000000
        min_index = square_distance.argmin(axis=1)
        logits = logits[min_index]

    mask = torch.ones_like(logits).scatter_(1, labels.unsqueeze(1), 0.)

    if dense:
        # start = time.time()
        dense_argmax = feats
        n1, c = logits.shape  # HACK
        n2, count2 = dense_argmax.shape
        assert n1 == n2
        scores=torch.zeros([n1,max(c,count2)])
        values=torch.ones_like(scores)
        logits=scores.scatter_add(1,dense_argmax,values)[:,:c]
        # end = time.time()
        # print("vote time:{}".format(end-start))
        
    rank_max = None
    if rank == 'soft':
        rank = soft_rank(-logits, regularization_strength=0.5)

    elif rank == 'hard':
        # zipf = 1 / (np.arange(c-1)+1)
        # zipf = zipf / zipf.sum()
        # res = logits[mask.bool()].view((logits.shape[0], logits.shape[1]-1))
        # ret = np.zeros(res.shape)
        # index = np.argsort(-res, axis=1)
        # for i in range(ret.shape[0]):
        #     ret[i][index[i]] = zipf

        # use rankdata
        rank = rankdata(-logits, axis=1, method='min')
        if distribution == "unbiased_zipf" or distribution == 'exponential_decay':
            rank_max = rankdata(-logits, axis=1, method='max')
    # if with_target:
    #     zipf = get_distribution_for_label_smooth(rank, distribution, power,dense,left_bound,mask,rank_max,right_bound)
    #     return Tensor(zipf.float())
    # else:
    #     ## new mask
    zipf = get_masked_distribution(args,
                                   rank, distribution, power, dense, left_bound, mask, rank_max, right_bound, with_target)

    # old mask
    # zipf = get_distribution(rank, distribution, power, dense,left_bound, rank_max)
    # zipf = zipf[mask.bool()].reshape((n, c-1))

    if isinstance(zipf, torch.Tensor):
        zipf = zipf.float()
    else:
        zipf = Tensor(zipf)

    return zipf


if __name__ == '__main__':
    import argparse
    from scipy.stats import rankdata
    parser = argparse.ArgumentParser()
    parser.add_argument("--smallest_prob", default=None, type=float, help="")
    parser.add_argument("--exp_lambda", default=None, type=float, help="")
    args = parser.parse_args()

    c = 5
    #logits = np.arange(c)[None, :] + 1
    logits = np.array([4, 3, 3, 2, 1])[None, :]
    rank = rankdata(-logits, axis=1, method='min')
    rank_max = rankdata(-logits, axis=1, method='max')
    dis_type = 'linear_decay'
    dist = gen_pdf_linear_decay(rank, c, 0, rank_max=rank_max)
    print(rank)
    print(rank_max)
    print(dist)

    dis_type = 'exponential_decay'
    rank_max = rank
    dist = gen_pdf_exponential_decay(
        rank, c, False, 0.01, 1, exp_lambda=args.exp_lambda, rank_max=rank_max)
    print(dist / dist.sum())
    dist = gen_pdf_exponential_decay(
        rank, c, False, 0.01, 1, exp_lambda=2., rank_max=rank_max)
    print(dist / dist.sum())
    dist = gen_pdf_exponential_decay(
        rank, c, False, 0.01, 1, exp_lambda=3., rank_max=rank_max)
    print(dist / dist.sum())
