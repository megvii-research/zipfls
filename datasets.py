import os
import megengine.data as data
import megengine.data.transform as T
from megengine.data.dataset.vision.folder import ImageFolder

def get_dataset(data_name, data_dir):
    train_dataset, test_dataset, num_classes = None, None, 100
    if data_name.lower() == 'cifar100':
        train_dataset = data.dataset.CIFAR100(root=data_dir, train=True)
        test_dataset = data.dataset.CIFAR100(root=data_dir, train=False)
        num_classes = 100
    elif data_name.lower() == 'imagenet':
        train_dataset = data.dataset.ImageNet(data_dir, train=True)
        test_dataset = data.dataset.ImageNet(data_dir, train=False)
        num_classes = 1000
    elif data_name.lower() in ['tinyimagenet', 'inat21']:
        '''
        The folder is expected to be organized as followed: data_dir/train(val)/cls/xxx.img_ext.
        Below is an example:
        tinyimagenet
        ├── train
        │   ├── n01443537
        │   ├── n01629819
        │   ├── n01641577
        │   ├── ......
        └── val
            ├── n01443537
            ├── n01629819
            ├── ......
        '''
        train_dataset = ImageFolder(root=os.path.join(data_dir, 'train'))
        test_dataset = ImageFolder(root=os.path.join(data_dir, 'val'))
        if data_name.lower() == 'tinyimagenet':
            num_classes = 200
        elif data_name.lower() == 'inat21':
            num_classes = 10000
    else: 
        raise NotImplementedError
    return train_dataset, test_dataset, num_classes

def load_dataset(name, data_dir, batch_size, workers):
    print('load dataset:', name)
    train_dataset, test_dataset, num_classes = get_dataset(name, data_dir)
    
    if name.lower().startswith('cifar'):
        train_transform = T.Compose([
            T.RandomCrop(32, padding_size=4),
            T.RandomHorizontalFlip(),
            T.Normalize(mean=(113.86, 122.96, 125.31), std=(51.26, 50.85, 51.59)),
            T.ToMode("CHW"),
        ])
        test_transform = T.Compose([
            T.Resize(32),
            T.Normalize(mean=(113.86, 122.96, 125.31), std=(51.26, 50.85, 51.59)),
            T.ToMode("CHW"),
        ])
    elif name.lower().startswith('tinyimagenet'):
        train_transform = T.Compose([
            T.RandomResizedCrop(32),
            T.RandomHorizontalFlip(),
            T.Normalize(mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]),
            T.ToMode("CHW"),
        ])
        test_transform = T.Compose([
            T.Resize(32),
            T.Normalize(mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]),
            T.ToMode("CHW"),
        ])
    elif name.lower() == 'imagenet':   
        train_transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.Normalize(
                mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]
            ),  # BGR
            T.ToMode("CHW"),
        ])
        test_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize(
                mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]
            ),  # BGR
            T.ToMode("CHW"),
        ])
    elif name.lower() == 'inat21':   
        train_transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.Normalize(
                mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]
            ),  # BGR
            T.ToMode("CHW"),
        ])
        test_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize(
                mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]
            ),  # BGR
            T.ToMode("CHW"),
        ])
    else:
        raise NotImplementedError('Not Defined this Dataset')

    train_sampler = data.RandomSampler(train_dataset, batch_size=batch_size)
    test_sampler = data.SequentialSampler(test_dataset, batch_size=batch_size)
    train_dataloader = data.DataLoader(train_dataset, train_sampler, train_transform, num_workers=workers)
    test_dataloader = data.DataLoader(test_dataset, test_sampler, test_transform, num_workers=workers)
    return train_dataloader, test_dataloader, num_classes