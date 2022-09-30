import numpy as np
from torch import Tensor
import cifar_models
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F


def densenet_train_forward(self, x: Tensor) -> Tensor:
    features = self.features(x)
    out = F.relu(features, inplace=True)
    featmap = out
    n, c, h, w = featmap.size()
    dense_logits_final = linear(featmap.permute(0, 2, 3, 1).reshape(
        n*h*w, c)).reshape(n, h*w, -1).argmax(axis=2)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    feat_final = out
    out = self.classifier(out)
    logit_final = out
    return [(logit_final, feat_final, dense_logits_final)]


def densenet_test_forward(self, x: Tensor) -> Tensor:
    features = self.features(x)
    out = F.relu(features, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    feat_final = out
    out = self.classifier(out)
    logit_final = out
    return [(logit_final, feat_final, None)]


def mobilenet_train_forward(self, x: Tensor) -> Tensor:
    x = self.features(x)
    featmap = x
    n, c, h, w = featmap.size()
    dense_logits_final = linear(featmap.permute(0, 2, 3, 1).reshape(
        n*h*w, c)).reshape(n, h*w, -1).argmax(axis=2)

    x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
    feat_final = x
    x = self.classifier(x)
    logit_final = x
    return [(logit_final, feat_final, dense_logits_final)]


def mobilenet_test_forward(self, x: Tensor) -> Tensor:
    x = self.features(x)
    x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
    feat_final = x
    x = self.classifier(x)
    logit_final = x
    return [(logit_final, feat_final, None)]


np.random.seed(1)
x = np.random.randn(3, 3, 224, 224)
# print(x)
label = [0, 1, 2]

x = Tensor(x)
label = Tensor(label).long()

# model_names = ['resnet18', 'resnet50', 'resnet101']
model_names = ['resnet18', 'resnet50', 'resnet101',
               'resnext50_32x4d', 'resnext101_32x8d', 'densenet121', 'MobileNetV2']
# model_names = ['densenet121', 'MobileNetV2']
model_names = ['densenet121']

for model_name in model_names:
    print(model_name)
    net = models.__dict__[model_name](
        num_classes=114)
    outs = net(x)
    for out in outs:
        for x in out:
            print(x.shape)
