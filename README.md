
This repo is the officialimplementation of the ECCV2022 paper: Efficient One Pass Self-distillation with Zipf's Label Smoothing. 

# [Zipf's LS: Efficient One Pass Self-distillation with Zipf's Label Smoothing](https://arxiv.org/abs/2207.12980)

## Framework & Comparison
<div style="text-align:center"><img src="megengine_zipfls/pics/framework.png" width="100%" ></div>
<div style="text-align:center"><img src="megengine_zipfls/pics/comparison.png" width="100%" ></div>

\[2022.9\] Pytorch Zipf's label smoothing is uploaded!

\[2022.7\] MegEngine Zipf's label smoothing is uploaded!

## Main Results
| Method              | DenseNet121   | DenseNet121  | ResNet18   | ResNet18     |
|:--------------------|:--------------|:-------------|:-----------|:-------------|
| **Arch**                | **CIFAR100**      | **TinyImageNet** | **CIFAR100**   | **TinyImageNet** |
| Pytorch Baseline    | 77.86±0.26    | 60.31±0.36   | 75.51±0.28 | 56.41±0.20   |
| **Pytorch Zipf's LS**   | **79.03±0.32**    | **62.64±0.30**   | **77.38±0.32** | **59.25±0.20**   |
| Megengine Baseline  | 77.97±0.18    | 60.78±0.31   | 75.29±0.29 | 56.03±0.34   |
| **Megengine Zipf's LS** | **79.85±0.27**    | **62.35±0.32**   | **77.08±0.28** | **59.01±0.23**   |

## Training
train_baseline_cifar100_resnet18:
```bash
python3 train.py --ngpus 1 --dataset CIFAR100 --data_dir cifar100_data --arch CIFAR_ResNet18 --loss_lambda 0.0 --alpha 0.0 --dense
```
train_ZipfsLS_cifar100_resnet18:
```bash
python3 train.py --ngpus 1 --dataset CIFAR100 --data_dir cifar100_data --arch CIFAR_ResNet18 --loss_lambda 0.1 --alpha 0.1 --dense
```
See more examples in [Makefile](megengine_zipfls/Makefile).
# Liscense
Zipf's LS is released under the Apache 2.0 license. See [LICENSE](LICENSE) for details.
