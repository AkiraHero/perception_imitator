import yaml


if __name__ == '__main__':

    '''
    for train
    '''
    outfile = "/home/xlju/Project/ModelSimulator/utils/config/dataset/minist.yaml"
    config_dict = {
        'dataset_class': 'MinistDataset',
        'paras': {
            'for_train': True,
            'batch_size': 16,
            'data_root': '/home/xlju/Project/ModelSimulator/data/MNIST/raw',
            'shuffle': True,
            'num_workers': 4
            }
        }

    with open(outfile, 'w') as f:
        yaml.dump(config_dict, f)


