import torch.nn as nn
import numpy as np
from torch import Tensor
from tqdm import tqdm
import torchvision.models as models
import torch
from scipy.stats import rankdata
from losses import zipf_loss, hard_friend_rank, soft_friend_rank, dense_rank, gen_pdf
import cifar_models as models
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def forward_with_feat(self, x: Tensor) -> Tensor:
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


def gen_pdf_from_friend(logits, feats, label):
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


net = models.load_model('resnet18', 100, pretrained=False)

np.random.seed(1)
x = np.random.randn(3, 3, 224, 224)
# print(x)
label = [0, 1, 2]

x = Tensor(x)
label = Tensor(label).long()

out = net(x)
feat, out = net(x)


print(out.shape, feat.shape)
loss = zipf_loss(out.cuda(), feat.cuda(), label.cuda(), loss_mask=False,
                 distribution='rank_zipf', dense=False, rank='hard',  friend=True)
print('loss', loss)

loss = hard_friend_rank(out.cuda(), feat.cuda(), label.cuda(), False)
print('loss', loss)

loss = zipf_loss(out.cuda(), feat.cuda(), label.cuda(), loss_mask=False,
                 distribution='rank_zipf', dense=False, rank='soft',  friend=True)
print('loss', loss)

loss = soft_friend_rank(out.cuda(), feat.cuda(), label.cuda(), False)
print('loss', loss)

loss = zipf_loss(out.cuda(), feat.cuda(), label.cuda(), loss_mask=False,
                 distribution='rank_zipf', dense=True, rank='hard',  friend=False)
print('loss', loss)

loss = dense_rank(out.cuda(), feat.cuda(), label.cuda(), False)
print('loss', loss)

# pdf = gen_pdf_from_friend(out, feat, label)
# print(pdf)
# pdf = gen_zipf_law_distribution_instance_wise(out, label)
# print(pdf)


dist = gen_pdf(out.cuda(), feat.cuda(), label.cuda(), loss_mask=False,
               distribution='rank_zipf', dense=False, rank='hard',  friend=False)
print('dist sum', dist.sum(dim=1))
plt.plot(np.sort(dist, axis=1)[0][::-1])
plt.savefig('./dists.png')
