import torch.nn

from model.basic_module.basic_module import BasicModule
import torch.nn as nn
import torch.nn.functional as func
from utils.model.model_utils import cal_conv2d_output_shape, cal_max_pool2d_output_shape, act_func_dict

'''
CNN
Config Parameters:
    [
        {
            name: 'conv2d',
            chn_in: 1,
            chn_out: 6,
            kernel_size: 3
            stride: 1,
            padding: 2,
            act_func: 'relu'
        },
        {
            name: 'linear',
            node_num: 5,
            act_func: 
        },
        {
            name: 'max_pool2d',
            kernel_size: 2,
            stride: 1
        }
    ]
'''


class CNN(BasicModule):
    def __init__(self, config):
        super(CNN, self).__init__()
        struct_list = config['paras']['struct_list']
        # shape should be channels * Height * Width
        input_shape = config['paras']['input_size']
        self._mod_list = []
        for unit_config in struct_list:
            obj, input_shape = CNN.build_unit(input_shape, unit_config)
            self._mod_list.append(obj)
        self.output_shape = input_shape

    def forward(self, x):
        for mod in self._mod_list:
            x = mod(x)
        return x

    @staticmethod
    def check_config(config):
        BasicModule.check_config(config)
        required_paras = ['input_size', 'struct_list']
        #  check necessary parameters
        BasicModule.check_config_dict(required_paras, config['paras'])

    @staticmethod
    def build_unit(input_shape, unit_config):
        func_dict = {
            'conv2d': CNN.build_conv2d,
            'linear': CNN.build_linear,
            'max_pool2d': CNN.build_max_pool2d
        }
        if unit_config['name'] not in func_dict.keys():
            raise KeyError
        return func_dict[unit_config['name']](input_shape, unit_config['paras'])

    @staticmethod
    def build_conv2d(input_shape, unit_config):
        if 3 != len(input_shape):
            raise TypeError(f'Require 3-dim shape, input is {len(input_shape)}')
        obj_list = []
        if input_shape[0] != unit_config['chn_in']:
            tmp = unit_config['chn_in']
            err = f'designated input channel {tmp} is not consistent with the actual input channel: {input_shape[0]}'
            raise TypeError(err)
        conv2d = nn.Conv2d(
            in_channels=unit_config['chn_in'],
            out_channels=unit_config['chn_out'],
            kernel_size=unit_config['kernel_size'],
            stride=unit_config['stride'],
            padding=unit_config['padding']
        )
        obj_list.append(conv2d)
        if unit_config['act_func'] != 'none':
            act_func = act_func_dict[unit_config['act_func']]
            obj_list.append(act_func())
        obj = torch.nn.Sequential(*obj_list)
        out_shape = cal_conv2d_output_shape(input_shape[1], input_shape[2], conv2d)
        return obj, (unit_config['chn_out'], *out_shape)

    @staticmethod
    def build_linear(input_shape, unit_config):
        obj_list = []
        input_size = 0
        if isinstance(input_shape, int):
            input_size = input_shape
        elif isinstance(input_shape, tuple) and 3 == len(input_shape):
            input_size = input_shape[0] * input_shape[1] * input_shape[2]
            view = nn.Flatten()
            obj_list.append(view)
        else:
            raise TypeError(f'input shape dim must be 1(int) or 3(chn-h-w)')
        linear = nn.Linear(input_size, unit_config['node_num'])
        obj_list.append(linear)
        if unit_config['act_func'] != 'none':
            act_func = act_func_dict[unit_config['act_func']]
            obj_list.append(act_func())
        obj = torch.nn.Sequential(*obj_list)
        return obj, unit_config['node_num']

    @staticmethod
    def build_max_pool2d(input_shape, unit_config):
        if 3 != len(input_shape):
            raise TypeError(f'Require 3-dim shape, input is {len(input_shape)}')
        obj = nn.MaxPool2d(stride=unit_config['stride'],
                           kernel_size=unit_config['kernel_size']
                           )
        out_shape = cal_max_pool2d_output_shape(input_shape[1], input_shape[2], obj)
        return obj, (input_shape[0], *out_shape)

    @staticmethod
    def build_module(config):
        CNN.check_config(config)
        return CNN(config)
