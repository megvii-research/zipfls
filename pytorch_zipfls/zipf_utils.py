import numpy as np
import torch.nn.functional as F
import torch
#from fast_soft_sort.pytorch_ops import soft_rank, soft_sort
# from torchsort import soft_rank, soft_sort
import torch.nn as nn
import torch.nn.functional as F
import torch
# from torchsort import soft_rank, soft_sort
import numpy as np
from scipy.stats import rankdata
from torch import Tensor
from collections import Counter

def softmax(array):
    maxv = array.max(axis=1, keepdims=True)
    array = np.exp(array - maxv)
    array = array / array.sum(axis=1, keepdims=True)
    return array


def remove_eye(A):
    return A[~np.eye(A.shape[0], dtype=bool)].reshape(A.shape[0], -1)


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


def gen_zipf_law_distribution_instance_wise(output, label):
    # input n, c
    # label n,
    # return n, c-1
    mask = torch.ones_like(output).scatter_(1, label.unsqueeze(1), 0.)
    res = output[mask.bool()].view((output.shape[0], output.shape[1]-1))

    res = res.detach().cpu().numpy()

    c = res.shape[1] + 1

    zip_dist = 1 / (np.arange(c-1)+1)
    zip_dist = zip_dist / zip_dist.sum()

    ret = np.zeros(res.shape)

    index = np.argsort(-res, axis=1)

    for i in range(ret.shape[0]):
        ret[i][index[i]] = zip_dist

    return ret


def gen_non_target_class_distribution(classwise_stats, label, dis_type):
    # classwise_stats c, c
    # label n,
    # return n, c-1
    n = label.shape[0]
    c = classwise_stats.shape[1]
    assert dis_type in ['rank_zipf', 'random_uniform',
                        'random_gaussian', 'random_pareto', 'constant']

    shape = (n, c-1)
    if dis_type == 'random_uniform':
        dist = np.random.rand(*shape)
    if dis_type == 'random_gaussian':
        dist = np.abs(np.random.randn(*shape))
    if dis_type == 'random_pareto':
        dist = np.random.pareto(1, shape)
    if dis_type == 'constant':
        dist = np.ones(shape)
    if dis_type == 'rank_zipf':
        classwise_stats = remove_eye(classwise_stats)
        index = np.argsort(-classwise_stats, axis=1)

        zipf = 1 / (np.arange(c-1)+1)
        dist = np.zeros(classwise_stats.shape)

        for i in range(c):
            dist[i][index[i]] = zipf

        dist = dist[label, :]

    dist = dist / dist.sum(axis=1, keepdims=True)

    return dist


def gen_pdf_from_friend_nondiff(logits, feats, label):
    n = logits.shape[0]
    feats = feats.detach()
    logits = logits.detach()

    A = (feats ** 2).sum(axis=1, keepdims=True)
    B = A.T
    dot = torch.matmul(feats, feats.T)

    square_distance = A + B - 2*dot
    square_distance[torch.arange(n), torch.arange(n)] = 1000000
    min_index = square_distance.argmin(axis=1)
    #print('min_index', min_index)
    #print('before logits', logits)
    logits = logits[min_index]
    #print('after logits', logits)
    # print(logits.shape)
    #
    mask = torch.ones_like(logits).scatter_(1, label.unsqueeze(1), 0.)
    res = logits[mask.bool()].view((logits.shape[0], logits.shape[1]-1))

    res = res.detach().cpu().numpy()

    c = 1000
    zip_dist = 1 / (np.arange(c-1)+1)
    zip_dist = zip_dist / zip_dist.sum()

    ret = np.zeros(res.shape)

    index = np.argsort(-res, axis=1)

    for i in range(ret.shape[0]):
        ret[i][index[i]] = zip_dist

    return ret


def gen_pdf_from_friend_soft(logits, feats, label):
    logits = logits.detach().cpu()
    feats = feats.detach().cpu()
    label = label.detach().cpu()
    n = logits.shape[0]
    n, c = logits.shape
    #feats = feats.detach()
    #logits = logits.detach()

    A = (feats ** 2).sum(axis=1, keepdims=True)
    B = A.T
    dot = torch.matmul(feats, feats.T)

    square_distance = A + B - 2*dot
    square_distance[torch.arange(n), torch.arange(n)] = 1000000
    min_index = square_distance.argmin(axis=1)

    rank = soft_rank(-logits, regularization_strength=0.5)

    zipf = 1 / rank
    zipf = zipf / zipf.sum(axis=1, keepdims=True)

    friend_zipf = zipf[min_index]
    mask = torch.ones((n, c)).scatter_(1, label.unsqueeze(1), 0.)
    res = friend_zipf[mask.bool()].view((n, c-1))

    return res


def gen_pdf_from_friend_hard(logits, feats, label):
    n = logits.shape[0]
    feats = feats.detach()
    logits = logits.detach()
    n, c = logits.shape

    A = (feats ** 2).sum(axis=1, keepdims=True)
    B = A.T
    dot = torch.matmul(feats, feats.T)

    square_distance = A + B - 2*dot
    square_distance[torch.arange(n), torch.arange(n)] = 1000000
    min_index = square_distance.argmin(axis=1)
    #print('min_index', min_index)
    #print('before logits', logits)
    logits = logits[min_index]
    #print('after logits', logits)
    # print(logits.shape)
    #
    mask = torch.ones_like(logits).scatter_(1, label.unsqueeze(1), 0.)
    res = logits[mask.bool()].view((logits.shape[0], logits.shape[1]-1))

    res = res.detach().cpu().numpy()

    zip_dist = 1 / (np.arange(c-1)+1)
    zip_dist = zip_dist / zip_dist.sum()

    ret = np.zeros(res.shape)

    index = np.argsort(-res, axis=1)

    for i in range(ret.shape[0]):
        ret[i][index[i]] = zip_dist

    return ret


def gen_pdf_from_dense_argmax(output, dense_argmax, label):
    output = output.detach().cpu()
    dense_argmax = dense_argmax.detach().cpu()
    label = label.detach().cpu()

    c = output.shape[1]  # HACK
    dense_argmax = dense_argmax.detach().cpu().numpy()
    n, count = dense_argmax.shape
    scores = np.zeros((n, c))
    for i in range(n):
        for j in range(count):
            scores[i][int(dense_argmax[i][j])] += 1

    # scores n,1000
    rank = rankdata(-scores, axis=1, method='min')
    zipf_dist = 1 / rank
    zipf_dist = zipf_dist / zipf_dist.sum(axis=1, keepdims=True)

    zipf_dist = Tensor(zipf_dist)
    mask = torch.ones_like(zipf_dist).scatter_(1, label.unsqueeze(1), 0.)
    zipf_dist = zipf_dist[mask.bool()].view((n, c-1))

    return zipf_dist

def gen_pdf_unbiased(rank,n_classes,dense=False,left_bound=0.01,right_bound=1.0,power=1.0,rank_max=None):
    interval=(right_bound-left_bound)/n_classes
    # print(interval)
    if dense:
        # rank_diff=(rank==rank_ord)
        # rank_diff=rank_diff.reshape(rank.shape)
        cnt=rank_max-rank+1
        if power!=1.0:
            dist=(1/(left_bound+(rank-1)*interval)**(power-1)-1/(left_bound+(rank+cnt-1)*interval)**(power-1))/cnt    #1/r**power integral
        else:
            dist=np.log((left_bound+(rank+cnt-1)*interval)/(left_bound+(rank-1)*interval))/cnt
    else:
        if power!=1.0:
            dist=1/(left_bound+(rank-1)*interval)**(power-1)-1/(left_bound+(rank)*interval)**(power-1)    #1/r**power integral
        else:
            dist=np.log((left_bound+(rank)*interval)/(left_bound+(rank-1)*interval))
    return dist

def gen_pdf_exponential_decay(rank,n_classes,dense=False,left_bound=0.01,right_bound=1.0,exp_lambda=None,rank_max=None):
    interval=(right_bound-left_bound)/n_classes
    # print(interval)
    def func(r):
        return np.exp(-exp_lambda * r)

    if dense:
        cnt=rank_max-rank+1
        r = left_bound+(rank+cnt-1)*interval
        l = left_bound+(rank-1)*interval
        dist = (func(r) - func(l)) / cnt
    else:
        r = left_bound+(rank)*interval
        l = left_bound+(rank-1)*interval
        dist = (func(r) - func(l))
    return dist

def gen_pdf_linear_decay(rank,n_classes,smallest_prob,rank_max=None):
    s = smallest_prob
    c = n_classes
    if rank_max is None:
        #cnt=rank_max-rank+1
        rank_max = rank
    cnt = rank_max - rank + 1
    #print(cnt)
    linear_gap = 2 * (1 - c * s) / ((c-1)*c)
    dist = (((c - rank) * linear_gap + s) + ((c -  (rank_max)) * linear_gap + s))  / 2
    #print(dist)

    return dist

if __name__=="__main__":
    logits=torch.rand([128,100])
    logits[:,:80]=0.0
    rank = rankdata(-logits, axis=1, method='min')
    rank_max=rankdata(-logits,axis=1,method="max")
    dist=gen_pdf_unbiased(rank,100,True,power=1.0,rank_max=rank_max)
    print(dist.sum(axis=1))
    dist=dist/dist.sum(axis=1,keepdims=True)
    print(dist)
