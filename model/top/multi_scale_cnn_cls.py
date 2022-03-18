from model.model_base import ModelBase
import factory.model_factory as ModelFactory
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleCNNCls(ModelBase):
    def __init__(self, config):
        super(MultiScaleCNNCls, self).__init__()
        self.backbone = ModelFactory.ModelFactory.get_model(config['paras']['submodules']['backbone'])
        self.head = ModelFactory.ModelFactory.get_model(config['paras']['submodules']['head'])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        prior = 0.01
        self.head.clshead.weight.data.fill_(-math.log((1.0-prior)/prior))
        self.head.clshead.bias.data.fill_(0)
        self.head.reghead.weight.data.fill_(0)
        self.head.reghead.bias.data.fill_(0)

    def forward(self, x):
        features = self.backbone(x)
        cls, reg = self.head(features)

        pred = torch.cat([cls, reg], dim=1)

        return pred