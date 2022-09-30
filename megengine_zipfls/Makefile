train_baseline_cifar100_resnet18:
	python3 train.py --ngpus 1 --dataset CIFAR100 --data_dir /data/PublicDatasets/cifar100 --arch CIFAR_ResNet18 --loss_lambda 0.0 --alpha 0.0 --dense --batch_size 128 --epochs 200 --desc baseline

train_ZipfsLS_cifar100_resnet18:
	python3 train.py --ngpus 1 --dataset CIFAR100 --data_dir /data/PublicDatasets/cifar100 --arch CIFAR_ResNet18 --loss_lambda 0.1 --alpha 0.1 --dense --batch_size 128 --epochs 200 --desc ZipfLS

train_baseline_cifar100_densenet121:
	python3 train.py --ngpus 1 --dataset CIFAR100 --data_dir /data/PublicDatasets/cifar100 --arch CIFAR_DenseNet121 --loss_lambda 0.0 --alpha 0.0 --dense --batch_size 128 --epochs 200 --desc baseline

train_ZipfsLS_cifar100_densenet121:
	python3 train.py --ngpus 1 --dataset CIFAR100 --data_dir /data/PublicDatasets/cifar100 --arch CIFAR_DenseNet121 --loss_lambda 0.1 --alpha 0.1 --dense --batch_size 128 --epochs 200 --desc ZipfLS


train_baseline_tinyimagenet_resnet18:
	python3 train.py --ngpus 1 --dataset TinyImagenet --data_dir /data/PublicDatasets/tinyimagenet --arch CIFAR_ResNet18 --loss_lambda 0.0 --alpha 0.0 --dense --batch_size 128 --epochs 200 --desc baseline

train_ZipfsLS_tinyimagenet_resnet18:
	python3 train.py --ngpus 1 --dataset TinyImagenet --data_dir /data/PublicDatasets/tinyimagenet --arch CIFAR_ResNet18 --loss_lambda 1.0 --alpha 0.5 --dense --batch_size 128 --epochs 200 --desc ZipfLS

train_baseline_tinyimagenet_densenet121:
	python3 train.py --ngpus 1 --dataset TinyImagenet --data_dir /data/PublicDatasets/tinyimagenet --arch CIFAR_DenseNet121 --loss_lambda 0.0 --alpha 0.0 --dense --batch_size 128 --epochs 200 --desc baseline

train_ZipfsLS_tinyimagenet_densenet121:
	python3 train.py --ngpus 1 --dataset TinyImagenet --data_dir /data/PublicDatasets/tinyimagenet --arch CIFAR_DenseNet121 --loss_lambda 1.0 --alpha 0.5 --dense --batch_size 128 --epochs 200 --desc ZipfLS


train_baseline_imagenet_resnet50:
	python3 train.py --dataset ImageNet --data_dir /data/PublicDatasets/ImageNet --arch resnet50 --loss_lambda 0.0 --alpha 0.0 --dense --ngpus 4 --batch_size 64 --epochs 100 --desc baseline

train_ZipfsLS_imagenet_resnet50:
	python3 train.py --dataset ImageNet --data_dir /data/PublicDatasets/ImageNet --arch resnet50 --loss_lambda 0.1 --alpha 0.0 --dense --ngpus 4 --batch_size 64 --epochs 100 --desc ZipfLS


train_baseline_inat_resnet50:
	python3 train.py --dataset inat --data_dir /data/PublicDatasets/inat --arch resnet50 --loss_lambda 0.0 --alpha 0.0 --dense --ngpus 4 --batch_size 64 --epochs 100 --desc baseline

train_ZipfsLS_inat_resnet50:
	python3 train.py --dataset inat --data_dir /data/PublicDatasets/inat --arch resnet50 --loss_lambda 1.0 --alpha 0.0 --dense --ngpus 4 --batch_size 64 --epochs 100 --desc ZipfLS

