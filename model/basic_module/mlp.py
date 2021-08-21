from basic_module import BasicModule
import torch.nn as nn
import torch.nn.functional as func

'''
MLP
Config Parameters:
    LayerNodeNums: []
    LayerActFuncs: []
'''


class MLP(BasicModule):
    def __init__(self, input_size_, layer_node_nums, layer_act_funcs):
        super(MLP, self).__init__()
        # # 初始化四层神经网络 两个全连接的隐藏层，一个输出层
        # self.fc1 = nn.Linear(input_size,200) # 第一个隐含层
        # self.fc2 = nn.Linear(200, 100) # 第二个隐含层
        # self.fc3 = nn.Linear(100, 1)  # 输出层
        assert isinstance(layer_node_nums, list)
        assert isinstance(layer_act_funcs, list)
        assert len(layer_node_nums) == len(layer_act_funcs)
        self.acts = layer_act_funcs
        self.layers = []
        input_size = input_size_
        for i in layer_node_nums:
            self.layers.append(nn.Linear(input_size, i))
            input_size = i

    def forward(self, input_):
        input_data = input_
        for layer, act in zip(self.layers, self.acts):
            input_data = layer(input_data)
            input_data = act(input_data)
        return input_data


        # # 前向传播， 输入值：din, 返回值 dout
        # din = din.view(-1, din.shape[2])    # 将一个多行的Tensor,拼接成一行
        # dout = self.fc1(din)
        # dout = func.relu(dout)  # 使用 relu 激活函数
        # dout = self.fc2(dout)
        # dout = func.relu(dout)
        # dout = self.fc3(dout)
        # dout = func.sigmoid(dout)
        # # dout = F.softmax(dout, dim=1) # 输出层使用 softmax 激活函数,这里输出层为1,因此不需要softmax,使用sigmoid
        # return dout
    @staticmethod
    def check_config(config):
        required_paras = ['INPUT_SIZE', 'LAYER_NODE_NUMS', 'LAYER_ACT_FUNCS']
        #  check necessary parameters
        BasicModule.check_config_dict(required_paras, config)

    @staticmethod
    def build_module(config):
        MLP.check_config(config)
        input_size = config['INPUT_SIZE']
        layer_node_nums = config['LAYER_NODE_NUMS']
        layer_act_funcs = []
        act_func_dict = {
            'RELU': func.relu,
            'LEAKY_RELU': func.leaky_relu,
            'SIGMOID': func.sigmoid,
            'TANH': func.tanh
        }
        for i in config['LAYER_ACT_FUNCS']:
            if i in act_func_dict.keys():
                layer_act_funcs.append(act_func_dict[i])
            else:
                raise KeyError
        return MLP(input_size, layer_node_nums, layer_act_funcs)
