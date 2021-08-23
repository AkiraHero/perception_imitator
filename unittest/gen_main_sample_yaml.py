import yaml


if __name__ == '__main__':

    '''
    for train
    '''
    outfile = "/home/xlju/Project/ModelSimulator/utils/config/main/train.yaml"
    config_dict = {
        'dataset': {
            'dataset_class': 'MinistDataset',
            'config_file': '/home/xlju/Project/ModelSimulator/utils/config/dataset/minist.yaml'
        },
        'model': {
            'model_class': 'CNN',
            'config_file': '/home/xlju/Project/ModelSimulator/utils/config/model/samples/cnn.yaml'
        },
        'training': {
            'epoch': 100,
            'batch_size': 16,
            'device': 0
        },
        'logging': {
            'path': '',
            'ckpt_eph_interval': 2,
            'suffix': ''
        }
    }
    with open(outfile, 'w') as f:
        yaml.dump(config_dict, f)


