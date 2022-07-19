from functools import partial
from numbers import Real
from typing import Any, Callable, Mapping, Sequence, Union
import math
import megengine.hub as hub
import megengine.module as M
import megengine.functional.nn as F
from megengine.functional import argmax

__all__ = ['ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'CIFAR_ResNet', 'CIFAR_ResNet18', 'CIFAR_ResNet34', 'CIFAR_ResNet10']


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return M.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return M.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(M.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = M.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = M.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(M.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = M.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = M.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PreActBlock(M.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = M.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = M.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.relu = M.ReLU()

        self.shortcut = M.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = M.Sequential(
                M.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(self.relu(self.bn2(out)))
        out += shortcut
        return out


class ResNet(M.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None, upsample=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = M.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = M.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = M.ReLU()
        self.maxpool = M.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = M.AdaptiveAvgPool2d((1, 1))
        self.fc = M.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, M.Conv2d):
                M.init.msra_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    fan_in, _ = M.init.calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    M.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, M.BatchNorm2d):
                M.init.ones_(m.weight)
                M.init.zeros_(m.bias)
            elif isinstance(m, M.Linear):
                M.init.msra_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = M.init.calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    M.init.uniform_(m.bias, -bound, bound)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    M.init.zeros_(m.bn3.weight)
                elif isinstance(m, BasicBlock):
                    M.init.zeros_(m.bn2.weight)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = M.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = M.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return M.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        def get_dense_info(featmap, linear):
            n, c, h, w = featmap.shape
            dense_logits = argmax(linear(featmap.transpose(0, 2, 3, 1).reshape(
                n*h*w, c)).reshape(n, h*w, -1), axis=2)
            feat = self.avgpool(featmap)
            feat = feat.reshape(feat.shape[0], -1)
            naive_logit = linear(feat)

            return naive_logit, feat, dense_logits

        logit_final, feat_final, dense_logits_final = get_dense_info(x, self.fc)


        return [(logit_final, feat_final, dense_logits_final)]


class CIFAR_ResNet(M.Module):
    def __init__(self, block, num_blocks, num_classes=10, bias=True, upsample=None):
        super(CIFAR_ResNet, self).__init__()
        self.upsample=upsample
        self.in_planes = 64
        self.conv1 = conv3x3(3, 64)
        self.bn1 = M.BatchNorm2d(64)
        self.relu = M.ReLU()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.gap = M.AvgPool2d(4)
        self.linear = M.Linear(512*block.expansion, num_classes, bias=bias)

        self.gap3 = M.AvgPool2d(8)
        self.linear3 = M.Linear(256, num_classes, bias=bias)

        self.gap2 = M.AvgPool2d(16)
        self.linear2 = M.Linear(128, num_classes, bias=bias)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return M.Sequential(*layers)

    def forward(self, x, lin=0, lout=5):
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out = self.layer4(out3)

        def get_dense_info(featmap, linear, gap):
            if hasattr(self, 'upsample') and self.upsample:
                tmp = F.interpolate(featmap, scale_factor=2., mode=self.upsample)
            else:
                tmp = featmap
            n, c, h, w = tmp.shape
            dense_logits = argmax(linear(tmp.transpose(0, 2, 3, 1).reshape(
                n*h*w, c)).reshape(n, h*w, -1), axis=2)

            feat = gap(featmap)
            feat = feat.reshape(featmap.shape[0], -1)

            naive_logit = linear(feat)

            return naive_logit, feat, dense_logits

        logit_2, feat_2, dense_logits_2 = get_dense_info(out2, self.linear2, self.gap2)
        logit_3, feat_3, dense_logits_3 = get_dense_info(out3, self.linear3, self.gap3)

        del out2
        del out3

        logit_final, feat_final, dense_logits_final = get_dense_info(out, self.linear, self.gap)


        return [(logit_final, feat_final, dense_logits_final), (logit_3, feat_3, dense_logits_3)]
        # return [(logit_final, feat_final, dense_logits_final)]

def CIFAR_ResNet10(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [1, 1, 1, 1], **kwargs)


def CIFAR_ResNet18(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2, 2, 2, 2], **kwargs)


def CIFAR_ResNet34(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [3, 4, 6, 3], **kwargs)


def resnet10(pretrained=False, **kwargs):
    """Constructs a ResNet-10 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnext50_32x4d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   groups=32, width_per_group=4, **kwargs)
    return model


def resnext101_32x8d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   groups=32, width_per_group=8, **kwargs)
    return model
