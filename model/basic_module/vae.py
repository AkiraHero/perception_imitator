from model.model_base import ModelBase
import model.ModelFactory as ModelFactory
import torch.nn as nn
import torch.nn.functional as F
import torch

class VAE(ModelBase):
    def __init__(self, config):
        super(VAE, self).__init__()
        self.encoder = ModelFactory.ModelFactory.get_model(config['paras']['submodules']['encoder'])
        self.mod_dict.add_module("encoder", self.encoder)
        self.mod_dict.add_module('fc21', nn.Linear(120, 20))
        self.mod_dict.add_module('fc22', nn.Linear(120, 20))
        self.mod_dict.add_module('fc3', nn.Linear(20, 100))
        self.mod_dict.add_module('fc4', nn.Linear(100, 10))

    def forward(self, x):
        x = self.encoder(x)
        mu = self.mod_dict['fc21'](x)
        log_var = self.mod_dict['fc22'](x)
        z = self.reparameterize(mu, log_var)
        x = F.relu(self.mod_dict['fc3'](z))
        x = self.mod_dict['fc4'](x)  # x为10个类别的得
        return x, mu, log_var

    def reparameterize(self, mu, log_var):    # 最后得到的是u(x)+sigma(x)*N(0,I)
        std = log_var.mul(0.5).exp_().to(self.device)  # e**(x/2)
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)   # 用正态分布填充eps
        eps.requires_grad = True
        return eps.mul(std).add_(mu)

    @staticmethod
    def check_config(config):
        pass

    @staticmethod
    def build_module(config):
        VAE.check_config(config)
        return VAE()
