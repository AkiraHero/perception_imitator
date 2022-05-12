from model.model_base import ModelBase
import factory.model_factory as ModelFactory
import torch.nn as nn
import torch.nn.functional as F
import torch

class PredictHead(ModelBase):
    def __init__(self, config):
        super(PredictHead, self).__init__()
        self._input_size = config['paras']['input_size']
        self._hidden_layer = config['paras']['hidden_layer']
        self._group_num = config['paras']['init_group_num']
        self._final_layer = config['paras']['fin_layer']

        self.init_fc = nn.Sequential(
            nn.Linear(self._input_size, self._hidden_layer),
            nn.ReLU(),
            nn.GroupNorm(self._group_num, self._hidden_layer)
        )

        self.res_fc1 = nn.Sequential(
            nn.Linear(self._hidden_layer, self._hidden_layer),
            nn.ReLU(),
            nn.GroupNorm(self._group_num, self._hidden_layer)
        )

        self.res_fc2 = nn.Sequential(
            nn.Linear(self._hidden_layer, self._hidden_layer),
            nn.ReLU(),
            nn.GroupNorm(self._group_num, self._hidden_layer)
        )

        self.final_fc = nn.Linear(self._hidden_layer, self._final_layer)

    def forward(self, x):
        x = self.init_fc(x)

        res_x1 = self.res_fc1(x)
        x = x + res_x1
        res_x2 = self.res_fc2(x)
        x = x + res_x2

        x = self.final_fc(x)

        return x

    @staticmethod
    def check_config(config):
        pass

    @staticmethod
    def build_module(config):
        PredictHead.check_config(config)
        return PredictHead()
