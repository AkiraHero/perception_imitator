import yaml


if __name__ == '__main__':

    '''
    for train
    '''
    outfile = "/home/xlju/Project/ModelSimulator/utils/config/main/train.yaml"
    config_dict = {
        'dataset': {
            'dataset_class': 'MinistDataset',
            'config_file': '/home/xlju/Project/ModelSimulator/utils/config/dataset/minist.yaml',
            'batch_size': 16,
        },
        'model': {
            'model_class': 'VAEGANModel',
            'config_file': '/home/xlju/Project/ModelSimulator/utils/config/model/samples/vae_gan.yaml'
        },
        'training': {
            'trainer_class': 'VAEGANTrainer',
            'epoch': 100,
            'device': 0,
            'optimizer': [
                {
                    'name': 'encoder_opt',
                    'type': 'Adam',
                    'paras': {
                        'lr': 0.0001,
                        'betas': [0.5, 0.999]
                    }
                },
                {
                    'name': 'discriminator_opt',
                    'type': 'Adam',
                    'paras': {
                        'lr': 0.0001,
                        'betas': [0.5, 0.999]
                    }
                }
            ]
        },
        'logging': {
            'path': '',
            'ckpt_eph_interval': 2,
            'suffix': ''
        }
    }
    with open(outfile, 'w') as f:
        yaml.dump(config_dict, f)

