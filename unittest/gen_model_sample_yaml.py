import yaml


if __name__ == '__main__':

    '''
    MLP
    '''
    outfile = "/home/xlju/Project/ModelSimulator/utils/config/model/samples/mlp.yaml"
    config_dict = {
        'name': 'mlp',
        'paras': {
            'input_size': 206,
            'layer_node_num': [200, 100, 1],
            'layer_act_func': ['relu', 'relu', 'sigmoid']
        }
    }
    with open(outfile, 'w') as f:
        yaml.dump(config_dict, f)

    '''
    CNN
    '''

    outfile = "/home/xlju/Project/ModelSimulator/utils/config/model/samples/cnn.yaml"
    config_dict = {
        'name': 'cnn',
        'paras': {
            'input_size': [],
            'struct_list': [
                    {
                        'name': 'conv2d',
                        'paras': {
                            'chn_in': 1,
                            'chn_out': 6,
                            'kernel_size': 3,
                            'stride': 1,
                            'padding': 2,
                            'act_func': 'relu'
                        }
                    },
                    {
                        'name': 'max_pool2d',
                        'paras': {
                            'kernel_size': 2,
                            'stride': 1
                        }
                    },
                    {
                        'name': 'conv2d',
                        'paras': {
                            'chn_in': 6,
                            'chn_out': 16,
                            'kernel_size': 5,
                            'stride': 1,
                            'padding': 0,
                            'act_func': 'relu'
                        }
                    },
                    {
                        'name': 'max_pool2d',
                        'paras': {
                            'kernel_size': 2,
                            'stride': 1
                        }
                    },
                    {
                        'name': 'linear',
                        'paras': {
                            'node_num': 120,
                            'act_func': 'relu'
                        }
                    },
                    {
                        'name': 'linear',
                        'paras': {
                            'node_num': 84,
                            'act_func': 'relu'
                        }
                    },
                    {
                        'name': 'linear',
                        'paras': {
                            'node_num': 10,
                            'act_func': 'none'
                        }
                    },
                ]
        }
    }
    with open(outfile, 'w') as f:
        yaml.dump(config_dict, f)


