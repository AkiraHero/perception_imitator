from model.basic_module.basic_module import BasicModule
import torch.nn as nn
from utils.model.model_utils import act_func_dict


class MLP(BasicModule):
    def __init__(self, config):
        super(MLP, self).__init__()
        input_size_, layer_node_nums, layer_act_funcs = \
            config['paras']['input_size'], config['paras']['layer_node_num'], config['paras']['layer_act_func']
        assert isinstance(layer_node_nums, list)
        assert isinstance(layer_act_funcs, list)
        assert len(layer_node_nums) == len(layer_act_funcs)
        self.acts = layer_act_funcs
        self.layers = []
        input_size = input_size_
        for i in layer_node_nums:
            self.layers.append(nn.Linear(input_size, i))
            input_size = i
        self.output_shape = self.layers[-1]

    def forward(self, input_):
        input_data = input_
        for layer, act in zip(self.layers, self.acts):
            input_data = layer(input_data)
            input_data = act(input_data)
        return input_data

    @staticmethod
    def check_config(config):
        BasicModule.check_config(config)
        required_paras = ['input_size', 'layer_act_func', 'layer_node_num']
        #  check necessary parameters
        BasicModule.check_config_dict(required_paras, config['paras'])

    @staticmethod
    def build_module(config):
        MLP.check_config(config)
        input_size = config['paras']['input_size']
        layer_node_nums = config['paras']['layer_node_num']
        layer_act_funcs = []
        for i in config['paras']['layer_act_func']:
            if i in act_func_dict.keys():
                layer_act_funcs.append(act_func_dict[i])
            else:
                err = f'Required activating func {i} does not exist.'
                raise KeyError(err)
        return MLP(config)
