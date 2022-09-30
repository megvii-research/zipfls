# Zipf's Label Smoothing as One Pass Self-distillation

Our pytorch implementation of [Zipf's Label Smoothing] on CIFAR100, TinyImageNet, ImageNet, INAT21(train_mini)

# Our experimental environments

- torch 1.8.1
- torchvision 0.9.1

# Dataset orgnization
```
datas
├── cifar-100-python
└── ILSVRC2012
    ├── train
    │   ├── n01443537
    │   ├── n01629819
    │   ├── n01641577
    │   ├── ......
    └── val
        ├── n01443537
        ├── n01629819
        ├── ......
└── tinyimagenet
    ├── train
    │   ├── n01443537
    │   ├── n01629819
    │   ├── n01641577
    │   ├── ......
    └── val
        ├── n01443537
        ├── n01629819
        ├── ......
└── inat21
    ├── train_mini
        ├── ......
    ├── val
        ├── ......
    ├── train_mini.json
    ├── val.json
```

# Small scale classification on CIFAR and TinyImageNet(Single GPU only)

## cifar100 with resnet18
```
# CE baseline
python3 train.py --dataset CIFAR100 --batch_size 128 --epochs 200 --arch CIFAR_ResNet18  --loss_lambda 0.0   --dense --desc small_scale.baseline  --upsample bilinear --alpha 0.1  
# LS
python3 train.py --dataset CIFAR100 --batch_size 128 --epochs 200 --arch CIFAR_ResNet18  --loss_lambda 0.0   --dense --desc small_scale.ls  --upsample None --alpha 0.1 --criterion label_smooth --smooth_ratio 0.1  
# ours ZLS
python3 train.py --dataset CIFAR100 --batch_size 128 --epochs 200 --arch CIFAR_ResNet18  --loss_lambda 0.1   --dense --desc small_scale.zls --upsample bilinear --alpha 0.1  
```

## cifar100 with densenet121
```
# CE baseline
python3 train.py --dataset CIFAR100 --batch_size 128 --epochs 200 --arch CIFAR_DenseNet121  --loss_lambda 0.0   --dense --desc small_scale.baseline  --upsample bilinear --alpha 0.1  
# LS
python3 train.py --dataset CIFAR100 --batch_size 128 --epochs 200 --arch CIFAR_DenseNet121  --loss_lambda 0.0   --dense --desc small_scale.ls  --upsample None --alpha 0.1 --criterion label_smooth --smooth_ratio 0.1 
# ours ZLS
python3 train.py --dataset CIFAR100 --batch_size 128 --epochs 200 --arch CIFAR_DenseNet121  --loss_lambda 0.1   --dense --desc small_scale.zls --upsample bilinear --alpha 0.1  
```
## tiny with resnet18
```
# CE baseline
python3 train.py --dataset TinyImageNet --batch_size 128 --epochs 200 --arch CIFAR_ResNet18  --loss_lambda 0.0 --criterion ce   --dense --desc small_scale.baseline  --upsample None --alpha 0.5  
# LS
python3 train.py --dataset TinyImageNet --batch_size 128 --epochs 200 --arch CIFAR_ResNet18  --loss_lambda 0.0 --criterion ce   --dense --desc small_scale.ls  --upsample None --alpha 0.5  --criterion label_smooth --smooth_ratio 0.1 
# ours ZLS
python3 train.py --dataset TinyImageNet --batch_size 128 --epochs 200 --arch CIFAR_ResNet18  --loss_lambda 1.0 --criterion ce   --dense --desc small_scale.zls --upsample bilinear --alpha 0.5  
```
## tiny with densenet121
```
# CE baseline
python3 train.py --dataset TinyImageNet --batch_size 128 --epochs 200 --arch CIFAR_DenseNet121  --loss_lambda 0.0 --criterion ce   --dense --desc small_scale.baseline  --upsample None --alpha 0.5  
# LS
python3 train.py --dataset TinyImageNet --batch_size 128 --epochs 200 --arch CIFAR_DenseNet121  --loss_lambda 0.0 --criterion ce   --dense --desc small_scale.ls  --upsample None --alpha 0.5  --criterion label_smooth --smooth_ratio 0.1 
# ours ZLS
python3 train.py --dataset TinyImageNet --batch_size 128 --epochs 200 --arch CIFAR_DenseNet121  --loss_lambda 1.0 --criterion ce   --dense --desc small_scale.zls --upsample bilinear --alpha 0.5  
```

# Large scale classification on ImageNet and INAT(4 GPUs)

## INAT with resnet50
```
# CE baseline
python3 train.py --ngpu 4 --dataset INAT21 --batch_size 256 --epochs 100 --arch resnet50   --dense --desc large_scale.baseline --alpha 0 --loss_lambda 0.0  -c --val_batch_size 64 
# LS
python3 train.py --ngpu 4 --dataset INAT21 --batch_size 256 --epochs 100 --arch resnet50   --dense --desc large_scale.ls --alpha 0 --loss_lambda 0.0  -c --val_batch_size 64 --criterion label_smooth --smooth_ratio 0.1 
# ours ZLS
python3 train.py --ngpu 4 --dataset INAT21 --batch_size 256 --epochs 100 --arch resnet50   --dense --desc large_scale.zls --alpha 0 --loss_lambda 1.0  -c --val_batch_size 64 
```
## Imagenet with resnet50
```
# CE baseline
python3 train.py --ngpu 4 --dataset ImageNet --batch_size 256 --epochs 100 --arch resnet50   --dense --desc large_scale.baseline --alpha 0 --loss_lambda 0.0  -c --val_batch_size 64 
# LS
python3 train.py --ngpu 4 --dataset ImageNet --batch_size 256 --epochs 100 --arch resnet50   --dense --desc large_scale.ls --alpha 0 --loss_lambda 0.0  -c --val_batch_size 64 --criterion label_smooth --smooth_ratio 0.1 
# ours ZLS
python3 train.py --ngpu 4 --dataset ImageNet --batch_size 256 --epochs 100 --arch resnet50   --dense --desc large_scale.zls --alpha 0 --loss_lambda 0.1  -c --val_batch_size 64 
```

## ImageNet and INAT with different architetures
```
# MODEL = ResNet18,ResNet50,ResNet101,ResNeXt50_32x4d,ResNeXt101_32x8d,DenseNet121,MobileNetV2
# INAT x MODEL
python3 train.py --ngpu 4 --dataset INAT21 --batch_size 256 --epochs 100 --arch ${MODEL}   --dense --desc large_scale --alpha 0 --loss_lambda 1.0  -c --val_batch_size 64 
# Imagenet x resnet50
python3 train.py --ngpu 4 --dataset ImageNet --resolution 224 --batch_size 256 --epochs 100 --arch ${MODEL}   --dense --desc large_scale --alpha 0 --loss_lambda 0.1  -c --val_batch_size 64 
```

## Comparision between distributions on different datasets
distribution = constant, random_uniform, random_pareto, linear_decay
### INAT
```
python3 train.py --ngpu 4 --dataset INAT21 --resolution 224 --batch_size 256 --epochs 100 --arch resnet50  --distribution constant --dense --desc dist_compare --alpha 0 --loss_lambda 1.0  -c --val_batch_size 64 
python3 train.py --ngpu 4 --dataset INAT21 --resolution 224 --batch_size 256 --epochs 100 --arch resnet50  --distribution random_uniform --dense --desc dist_compare --alpha 0 --loss_lambda 1.0  -c --val_batch_size 64 
python3 train.py --ngpu 4 --dataset INAT21 --resolution 224 --batch_size 256 --epochs 100 --arch resnet50  --distribution random_pareto --dense --desc dist_compare --alpha 0 --loss_lambda 1.0  -c --val_batch_size 64 
python3 train.py --ngpu 4 --dataset INAT21 --resolution 224 --batch_size 256 --epochs 100 --arch resnet50  --distribution linear_decay --dense --desc dist_compare --alpha 0 --loss_lambda 1.0  -c --val_batch_size 64 --smallest_prob 0.000001 
```
### ImageNet
```
python3 train.py --ngpu 4 --dataset ImageNet --resolution 224 --batch_size 256 --epochs 100 --arch resnet50  --distribution constant --dense --desc 1108_distribution_compare --alpha 0 --loss_lambda 0.1  -c --val_batch_size 64 
python3 train.py --ngpu 4 --dataset ImageNet --resolution 224 --batch_size 256 --epochs 100 --arch resnet50  --distribution random_uniform --dense --desc 1108_distribution_compare --alpha 0 --loss_lambda 0.1  -c --val_batch_size 64 
python3 train.py --ngpu 4 --dataset ImageNet --resolution 224 --batch_size 256 --epochs 100 --arch resnet50  --distribution random_pareto --dense --desc 1108_distribution_compare --alpha 0 --loss_lambda 0.1  -c --val_batch_size 64 
python3 train.py --ngpu 4 --dataset ImageNet --resolution 224 --batch_size 256 --epochs 100 --arch resnet50  --distribution linear_decay --smallest_prob 0.000001 --dense --desc 1108_distribution_compare --alpha 0 --loss_lambda 0.1  -c --val_batch_size 64 
```
