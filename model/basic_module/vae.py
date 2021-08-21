# 生成器结构（暂时直接从CNN_Mnist中调用Net，不用这个）
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3, 1, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc21 = nn.Linear(120, 20)
        self.fc22 = nn.Linear(120, 20)
        self.fc3 = nn.Linear(20, 100)
        self.fc4 = nn.Linear(100, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))

        mu = self.fc21(x)
        logvar = self.fc22(x)

        z = self.reparametrize(mu,logvar)

        x = F.relu(self.fc3(z))
        x = self.fc4(x)# x为10个类别的得分
        # x = F.sigmoid(x)
        return x, mu, logvar

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def reparametrize(self, mu, logvar):    # 最后得到的是u(x)+sigma(x)*N(0,I)
        std = logvar.mul(0.5).exp_() # e**(x/2)
        eps = torch.FloatTensor(std.size()).normal_().cuda()   # 用正态分布填充eps
        if torch.cuda.is_available():
            eps = Variable(eps.cuda())
        else:
            eps = Variable(eps)
        return eps.mul(std).add_(mu)

    @staticmethod
    def check_config(config):
        pass

    @staticmethod
    def build_module(config):
        VAE.check_config(config)
        return VAE()
