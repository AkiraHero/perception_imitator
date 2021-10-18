from model.model_base import ModelBase
import factory.model_factory as ModelFactory
import torch.nn as nn
import torch.nn.functional as F
import torch


class VAEFP(ModelBase):
    def __init__(self, config):
        super(VAEFP, self).__init__()
        self.encoder = ModelFactory.ModelFactory.get_model(config['paras']['submodules']['encoder'])
        self.fc21 = nn.Linear(50, 20)
        self.fc22 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 200)
        self.fc4 = nn.Linear(200, 100)
        self.fc5 = nn.Linear(100, 7)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc21(x)
        log_var = self.fc22(x)
        z = self.reparameterize(mu, log_var)
        x = F.relu(self.fc3(z))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)     # 生成的具有fp特性的bbox七个参数，暂时去除类别
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
        VAEFP.check_config(config)
        return VAEFP()
