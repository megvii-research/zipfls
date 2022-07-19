import math
from functools import partial
from numbers import Real
from typing import Any, Callable, Mapping, Sequence, Union

import megengine.hub as hub
import megengine.module as M
import megengine.functional.nn as F
from megengine.functional import argmax


class Bottleneck(M.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = M.BatchNorm2d(in_planes)
        self.conv1 = M.Conv2d(in_planes, 4*growth_rate,
                               kernel_size=1, bias=False)
        self.bn2 = M.BatchNorm2d(4*growth_rate)
        self.conv2 = M.Conv2d(4*growth_rate, growth_rate,
                               kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = F.concat([out, x], 1)
        return out


class Transition(M.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = M.BatchNorm2d(in_planes)
        self.conv = M.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class CIFAR_DenseNet(M.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, bias=True):
        super(CIFAR_DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = M.Conv2d(
            3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes
        num_planes2 = num_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes
        num_planes3 = num_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = M.BatchNorm2d(num_planes)
        self.linear = M.Linear(num_planes, num_classes, bias=bias)

        self.gap = M.AvgPool2d(4)
        self.gap2 = M.AvgPool2d(8)
        self.linear2 = M.Linear(num_planes2, num_classes, bias=bias)
        self.bn2 = M.BatchNorm2d(num_planes2)
        self.gap3 = M.AvgPool2d(4)
        self.linear3 = M.Linear(num_planes3, num_classes, bias=bias)
        self.bn3 = M.BatchNorm2d(num_planes3)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return M.Sequential(*layers)

    # def forward(self, x):
    #     out = self.conv1(x)
    #     out = self.trans1(self.dense1(out))
    #     out = self.trans2(self.dense2(out))
    #     out = self.trans3(self.dense3(out))
    #     out = self.dense4(out)
    #     out = F.avg_pool2d(F.relu(self.bn(out))), 4)
    #     out=out.view(out.size(0), -1)
    #     prob=self.linear(out)
    #     return out, prob

    def forward(self, x, lin=0, lout=5):
        out = self.conv1(x)
        out1 = self.trans1(self.dense1(out))
        out2 = self.trans2(self.dense2(out1))
        out3 = self.trans3(self.dense3(out2))
        out = self.dense4(out3)
        # out = F.relu(self.bn(out))

        # print(out.shape, out2.shape, out3.shape)

        def get_dense_info(featmap, linear, gap, bn):
            # print('featmap', featmap.shape)
            featmap = F.relu(bn(featmap))
            if hasattr(self, 'upsample') and self.upsample:
                tmp = F.upsample(featmap, scale_factor=2.,
                                 mode=self.upsample)
            else:
                tmp = featmap
            n, c, h, w = tmp.shape
            dense_logits_argmax = argmax(linear(tmp.transpose(0, 2, 3, 1).reshape(
                n*h*w, c)).reshape(n, h*w, -1), axis=2)

            feat = gap(featmap)
            feat = feat.reshape(featmap.shape[0], -1)

            naive_logit = linear(feat)

            return naive_logit, feat, dense_logits_argmax

        logit_2, feat_2, dense_logits_2 = get_dense_info(
            out2, self.linear2, self.gap2, self.bn2)
        logit_3, feat_3, dense_logits_3 = get_dense_info(
            out3, self.linear3, self.gap3, self.bn3)

        del out2
        del out3

        logit_final, feat_final, dense_logits_final = get_dense_info(
            out, self.linear, self.gap, self.bn)

        return [(logit_final, feat_final, dense_logits_final), (logit_3, feat_3, dense_logits_3), (logit_2, feat_2, dense_logits_2)]


def CIFAR_DenseNet121(pretrained=False, num_classes=10, bias=True, **kwargs):
    return CIFAR_DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, num_classes=num_classes, bias=bias)


