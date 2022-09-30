import megengine 
import megengine.functional as F
import numpy as np
from megengine import Tensor

def zipf_loss(logits, feats, labels, loss_mask=False, dense=True):
    non_target_pdf = gen_pdf(logits, feats, labels, dense=dense)
    dist_loss = kl_loss(logits, labels, non_target_pdf, do_mask=loss_mask)
    return dist_loss

def kl_div(p, q):
    return q * (F.log(q)-F.logsoftmax(p, axis=1))

def kl_loss(output, label, pdf, do_mask=False):
    mask = F.ones_like(output)
    tmp = F.expand_dims(label, axis=1)
    F.scatter(mask, 1, tmp, F.zeros_like(tmp).astype('Float32'))
    res = output[mask.astype(np.bool)].reshape((output.shape[0], output.shape[1]-1))
    res = F.logsoftmax(res, axis=1)
    loss = kl_div(res, pdf).sum(axis=1)

    if do_mask:
        loss_mask = (F.argmax(output, 1) == label)
        if loss_mask.any():
            loss = loss * loss_mask
            loss = loss.sum() / loss_mask.sum()
        else:
            loss = loss.mean() * 0
    else:
        loss = loss.mean()

    return loss

def gen_pdf(logits, feats, labels, dense=True):
    n = logits.shape[0]
    n, c = logits.shape
    
    mask = F.ones_like(logits)
    tmp = F.expand_dims(labels, axis=1)
    F.scatter(mask, 1, tmp, F.zeros_like(tmp).astype('Float32'))
    
    if dense:
        d_n, d_count = feats.shape
        assert n == d_n
        feats = F.expand_dims(feats, axis=2).astype(np.int32)
        logits = F.zeros((n,d_count,c)).astype(np.int32)
        F.scatter(logits, 2, feats, F.ones_like(feats).astype(np.int32))
        logits = logits.sum(1)[:,:c]
   
    rank = rank_data(-logits)

    # zipf dist
    power = 1.0
    dist = (1 / rank) ** power
    dist = dist[mask.astype('bool')].reshape((n, c-1))
    n, c = rank.shape
    dist = dist / dist.sum(axis=1, keepdims=True)
    return dist

def rank_data(x):
    device = x.device
    vals,inds=F.sort(x)
    vals
    rank=F.zeros(x.shape,device=device).astype(np.int32)
    temp_inds = F.arange(x.shape[1]+1,dtype=np.int32,device=device)
    temp_inds = F.expand_dims(temp_inds, axis=0)
    temp_inds = F.broadcast_to(temp_inds, [x.shape[0],x.shape[1]+1])
    F.scatter(rank,1,inds,temp_inds[:,:-1].astype(np.int32))
    obs=(vals[:,1:]!=vals[:,:-1])
    all_true = F.ones([x.shape[0],1],dtype=bool,device=device)
    obs=F.concat([all_true,obs],axis=1).astype(np.int32)
    obs_cum=F.cumsum(obs,axis=1)
    obs_cum_mask=obs_cum*obs
    temp3=F.zeros(temp_inds.shape,dtype=np.int32,device=device)
    temp_inds, obs_cum_mask
    F.scatter(temp3, 1,obs_cum_mask,temp_inds[:,:-1])
    dense=F.gather(F.cumsum(obs, axis=1),1,rank)
    rank_data=F.gather(temp3,1,dense) + 1
    return rank_data


