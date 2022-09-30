#!/usr/bin/env python3
import torch
from functools import lru_cache
from torch.utils.data import Dataset, DataLoader
import pickle
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict
from sampler import FriendSampler
import json
import os
import numpy as np
from torch.utils import data
import torchvision
import torchvision.datasets
import io


class ImageNetDataset(Dataset):
    def __init__(self, train, transform=None):
        """
        Args:
            train (bool, optional): If True, creates dataset from training set, otherwise
                creates from test set.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train = train
        self.transform = transform
        self.samples = self.get()
        self.num_classes = 1000
        self.cls_idxs = defaultdict(list)
        for idx, (nid, label) in enumerate(self.samples):
            self.cls_idxs[label].append(idx)

    def get(self):
        return [None]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return None, None
        nid, label = self.samples[idx]
        img = get_img(nid)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class TinyImageNetDataset(Dataset):
    def __init__(self, train, transform=None):
        self.train = train
        self.transform = transform
        self.samples = self.get()
        self.num_classes = 200
        self.cls_idxs = defaultdict(list)
        for idx, (nid, label) in enumerate(self.samples):
            self.cls_idxs[label].append(idx)

    def get(self):
        return [None]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return None, None
        nid, label = self.samples[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class INAT21Dataset(Dataset):
    def __init__(self, state, transform=None):
        """
        Args:
            train (bool, optional): If True, creates dataset from training set, otherwise
                creates from test set.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_prefix = './datas/inat21/'
        self.state = state
        self.transform = transform
        self.samples, self.num_classes = self.get()

    def get(self):
        if self.state == 'train':
            jpath = self.data_prefix + 'train_mini.json'
        else:
            jpath = self.data_prefix + 'val.json'
        with open(jpath, 'r') as f:
            annotations = json.load(f)
        imgid2path = {i['id']: self.data_prefix + i['file_name'] for i in annotations['images']}
        imgid2class = {i['image_id']: i['category_id']
                       for i in annotations['annotations']}
        samples = [(imgid2path[i], imgid2class[i]) for i in imgid2path.keys()]
        num_classes = 10000
        return samples, num_classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class DatasetWrapper2():
    def __init__(self, state, data_dir, train_list_file, val_list_file, num_classes, transform=None):
        self.state = state
        self.transform = transform
        self.data_dir = data_dir
        self.train_list_file = train_list_file
        self.val_list_file = val_list_file
        self.num_classes = num_classes
        self.samples = self.get()

    def get(self):
        if self.state == 'train':
            fpath = self.train_list_file
        else:
            fpath = self.val_list_file

        img_path_labels = []
        with open(fpath, 'r') as f:
            for line in f:
                pic_name, label = line.strip()
                img_path = os.path.join(self.data_dir, pic_name)
                img_path_labels.append([img_path, label])

        return img_path_labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class DatasetWrapper(Dataset):
    # Additinoal attributes
    # - indices
    # - classwise_indices
    # - num_classes
    # - get_class

    def __init__(self, dataset, indices=None):
        self.base_dataset = dataset
        if indices is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = indices

        # torchvision 0.2.0 compatibility
        if torchvision.__version__.startswith('0.2'):
            if isinstance(self.base_dataset, datasets.ImageFolder):
                self.base_dataset.targets = [s[1]
                                             for s in self.base_dataset.imgs]
            else:
                if self.base_dataset.train:
                    self.base_dataset.targets = self.base_dataset.train_labels
                else:
                    self.base_dataset.targets = self.base_dataset.test_labels

        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.base_dataset.targets[self.indices[i]]
            self.classwise_indices[y].append(i)
        self.num_classes = max(self.classwise_indices.keys())+1

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, i):
        return self.base_dataset.targets[self.indices[i]]


def load_dataset(name, data_dir, batchsize=64, workers=4, train_sampler_method=None, val_batchsize=None, **kwargs):
    assert name in ['CIFAR100', 'ImageNet', 'TinyImageNet', 'INAT21']
    if name.startswith('CIFAR'):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True,
                                                      download=True, transform=transform_train)

        if not hasattr(train_dataset, 'num_classes'):
            setattr(train_dataset, 'num_classes', 100)
        val_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False,
                                                    download=True, transform=transform_test)

    elif name == 'ImageNet':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(
                224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
        ])
        transform_test = transforms.Compose([
                                            transforms.Resize(
                                                256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                                0.229, 0.224, 0.225]),
                                            ])
        train_val_dataset_dir = os.path.join(data_dir, "ILSVRC2012/train")
        test_dataset_dir = os.path.join(data_dir, "ILSVRC2012/val")

        train_dataset = DatasetWrapper(torchvision.datasets.ImageFolder(
            root=train_val_dataset_dir, transform=transform_train))
        val_dataset = DatasetWrapper(torchvision.datasets.ImageFolder(
            root=test_dataset_dir, transform=transform_test))

    elif name == 'TinyImageNet':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(
                32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
        ])
        transform_test = transforms.Compose([
                                            transforms.Resize(32),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                                0.229, 0.224, 0.225]),
                                            ])
        train_val_dataset_dir = os.path.join(data_dir, "tinyimagenet/train")
        test_dataset_dir = os.path.join(data_dir, "tinyimagenet/val")

        train_dataset = DatasetWrapper(torchvision.datasets.ImageFolder(
            root=train_val_dataset_dir, transform=transform_train))
        val_dataset = DatasetWrapper(torchvision.datasets.ImageFolder(
            root=test_dataset_dir, transform=transform_test))
    elif name == 'INAT21':
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_dataset = INAT21Dataset(state='train',
                                      transform=transforms.Compose([
                                            transforms.RandomResizedCrop(
                                                224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            normalize,
                                      ]))
        val_dataset = INAT21Dataset(state='val', transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    if train_sampler_method == 'friend':
        train_sampler = FriendSampler(train_dataset, 1)
    else:
        train_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=(
        train_sampler is None), num_workers=workers, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batchsize if val_batchsize is None else val_batchsize,
                            shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


if __name__ == '__main__':
    datasets_name = ['CIFAR10', 'CIFAR100',
                     'ImageNet', 'TinyImageNet', 'INAT21']
    datasets_name = ['CIFAR100', 'TinyImageNet', 'INAT21']
    datasets_name = ['ImageNet']
    for dataset_name in datasets_name:
        print('\nTesting {} data provider...'.format(dataset_name))
        # train_loader, val_loader = load_dataset(dataset_name)
        batchsize = 128
        train_loader, val_loader = load_dataset(
            dataset_name, './datas', batchsize=batchsize, train_sampler_method=None, resolution=32)
        # print('num_classes:', train_loader.dataset.num_classes)
        iterations = len(train_loader) / batchsize
        print('dataset size: ', len(train_loader), ' iterations:', iterations)
        for batch, target in train_loader:
            print('train_loader', batch.shape, target.shape)
            break
        for batch, target in val_loader:
            print('val_loader', batch.shape, target.shape)
            break
