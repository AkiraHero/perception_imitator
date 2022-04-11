from model.model_base import ModelBase
import factory.model_factory as ModelFactory
import torch.nn as nn
import torch.nn.functional as F
import torch


class VAEForCNN(ModelBase):
    def __init__(self, config):
        super(VAEForCNN, self).__init__()
        self.input_channel=192
        self.output_channel=1

        self.encoder = ModelFactory.ModelFactory.get_model(config['paras']['submodules']['encoder'])
        self.encode_2_one_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channel, out_channels=self.output_channel, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.fc21 = nn.Linear(8800, 512)
        self.fc22 = nn.Linear(8800, 512)
        self.fc3 = nn.Linear(512, 8800)
        self.upsample = nn.Sequential(     # 1*H/4*W/4 -> 1*H*W
            nn.ConvTranspose2d(in_channels=self.output_channel, out_channels=self.output_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.output_channel),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.output_channel, out_channels=self.output_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.output_channel),
            nn.ReLU()
        )
        self.clshead = nn.Conv2d(in_channels=self.output_channel, out_channels=self.output_channel, kernel_size=1)
        

    def forward(self, x):
        x = self.encoder(x)
        x = self.encode_2_one_channel(x)
        ori_shape = x.shape
        x = x.view(ori_shape[0], -1)
        mu = self.fc21(x)
        log_var = self.fc22(x)
        z = self.reparameterize(mu, log_var)
        x = F.relu(self.fc3(z))
        x = x.view(ori_shape)
        x = self.upsample(x)
        x = self.clshead(x)
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
        VAEForCNN.check_config(config)
        return VAEForCNN()
