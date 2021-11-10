from model.model_base import ModelBase
import factory.model_factory as ModelFactory
import torch.nn as nn
import torch.nn.functional as F
import torch


class VAEFPCloudPoint(ModelBase):
    def __init__(self, config):
        super(VAEFPCloudPoint, self).__init__()
        self.conv1 = nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc21 = nn.Linear(1024, 50)
        self.fc22 = nn.Linear(1024, 50)
        self.fc3 = nn.Linear(50, 200)
        self.fc4 = nn.Linear(200, 100)
        self.fc5 = nn.Linear(100, 7)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 调整通道可以进行conv1d卷积操作
        x = F.relu(self.bn1(self.conv1(x))) # 将每个点由4维升为64维
        x = F.relu(self.bn2(self.conv2(x)))  # 第二次mlp的第一层，64->128
        x = self.bn3(self.conv3(x))         # 第二次mlp的第一层，128->1024
        x = torch.max(x, 2, keepdim=True)[0]    # 最大池化操作，为pointnet核心操作
        x = x.view(-1, 1024)    # 得到batchsize*1024特征
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
        VAEFPCloudPoint.check_config(config)
        return VAEFPCloudPoint()
