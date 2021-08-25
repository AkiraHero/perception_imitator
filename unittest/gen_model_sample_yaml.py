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
            'input_size': [1, 100, 100],
            'struct_list': [
                    {
                        'class': 'conv2d',
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
                        'class': 'max_pool2d',
                        'paras': {
                            'kernel_size': 2,
                            'stride': 2
                        }
                    },
                    {
                        'class': 'conv2d',
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
                        'class': 'max_pool2d',
                        'paras': {
                            'kernel_size': 2,
                            'stride': 2
                        }
                    },
                    {
                        'class': 'linear',
                        'paras': {
                            'node_num': 120,
                            'act_func': 'relu'
                        }
                    },
                    {
                        'class': 'linear',
                        'paras': {
                            'node_num': 84,
                            'act_func': 'relu'
                        }
                    },
                    {
                        'class': 'linear',
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


    '''
    vae
    '''
    vae_gan_config_dict = {
        'name': 'vae',
        'paras': {
            'submodules': {
                'encoder': {
                    'model_class': 'CNN',
                    'config_file': '/home/xlju/Project/ModelSimulator/utils/config/model/samples/cnn_for_vae.yaml'
                }
            },
            'latent_distribution': 'Gaussian'
        }
    }
    outfile = "/home/xlju/Project/ModelSimulator/utils/config/model/samples/vae.yaml"
    with open(outfile, 'w') as f:
        yaml.dump(vae_gan_config_dict, f)






    '''
    vae-gan
    '''
    vae_gan_config_dict = {
        'name': 'VAEGANModel',
        'paras': {
            'submodules': {
                'encoder': {
                    'model_class': 'VAE',
                    'config_file': '/home/xlju/Project/ModelSimulator/utils/config/model/samples/vae.yaml'
                },
                'discriminator': {
                    'model_class': 'MLP',
                    'config_file': '/home/xlju/Project/ModelSimulator/utils/config/model/samples/mlp.yaml'
                },
                'target_model': {
                    'model_class': 'CNN',
                    'config_file': '/home/xlju/Project/ModelSimulator/utils/config/model/samples/cnn.yaml',
                    'model_para_file': '/home/xlju/Project/ModelSimulator/results/Mnist/param_minist_5.pt'
                }
            }
        }
    }
    outfile = "/home/xlju/Project/ModelSimulator/utils/config/model/samples/vae_gan.yaml"
    with open(outfile, 'w') as f:
        yaml.dump(vae_gan_config_dict, f)


