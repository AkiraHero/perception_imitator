from model.model_base import ModelBase
import factory.model_factory as ModelFactory
import torch.nn as nn
import torch.nn.functional as F
import torch


class VAEAllFPExplicit(ModelBase):
    def __init__(self, config):
        super(VAEAllFPExplicit, self).__init__()
        self.encoder = ModelFactory.ModelFactory.get_model(config['paras']['submodules']['encoder'])
        self.fc21 = nn.Linear(1024, 128)
        self.fc22 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 140)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc21(x)
        log_var = self.fc22(x)
        z = self.reparameterize(mu, log_var)
        x = F.relu(self.fc3(z))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)     # 生成一个场景中所有具有fp特性的bbox（个数上限为4个），暂时去除类别
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
        VAEAllFPExplicit.check_config(config)
        return VAEAllFPExplicit()
