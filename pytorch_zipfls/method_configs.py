method_params = {
    'tf-kd': {
        'CIFAR_ResNet18':
        {
            "reg_temperature": 20,
            # loss_lambda
            "reg_alpha": 0.1,
            "multiplier": 100.0,
        },
        'CIFAR_DenseNet121':
        {
            "reg_temperature": 40,
            "reg_alpha": 0.1,
            "multiplier": 1.0,
        },
        'resnet50':
        {
            "reg_temperature": 20.0,
            "reg_alpha": 0.1,
            "multiplier": 1.0,
        },
    },

    'cs-kd': {
        'CIFAR_ResNet18':
        {
            "temp_factor": 4.0,
            'loss_lambda': 1.0,
        },
        'CIFAR_DenseNet121':
        {
            "temp_factor": 4.0,
            'loss_lambda': 1.0,
        },
        'resnet50':
        {
            "temp_factor": 4.0,
            'loss_lambda': 3.0,
        },
    },

    'bake': {
        # cifar100
        # 'CIFAR_ResNet18':
        # {
        #     "temp_factor": 4.0,
        #     'intra-imgs': 3,
        #     'omega': 0.5,
        #     'loss_lambda': 1.0,
        # },
        # tiny
        'CIFAR_ResNet18':
        {
            "temp_factor": 4.0,
            'intra-imgs': 1,
            'omega': 0.9,
            'loss_lambda': 1.0,
        },
        'CIFAR_DenseNet121':
        {
            "temp_factor": 4.0,
            'intra-imgs': 1,
            'omega': 0.9,
            'loss_lambda': 1.0,
        },
        'resnet50':
        {
            "temp_factor": 4.0,
            'intra-imgs': 1,
            'omega': 0.5,
            'loss_lambda': 1.0,
        },
    },

}
