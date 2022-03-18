from model.model_base import ModelBase
import factory.model_factory as ModelFactory
import torch.nn as nn
import torch.nn.functional as F
import torch

class MultiScaleCNN(ModelBase):
    def __init__(self, config):
        super(MultiScaleCNN, self).__init__()
        self.in_ch = config['paras']['input_channel']
        self.out_ch = config['paras']['one_feature_channel']

        self.downsample1 = nn.Sequential(     # 2*H*W -> 64*H/4*W/4
            nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=5, stride=4, padding=1),
            nn.BatchNorm2d(self.out_ch),
            nn.ReLU(),
        )
        self.downsample2 = nn.Sequential(     # 64*H/4*W/4 -> 128*H/8*W/8
            nn.Conv2d(in_channels=self.out_ch, out_channels=self.out_ch*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.out_ch*2),
            nn.ReLU(),
        )
        self.downsample3 = nn.Sequential(     # 128*H/8*W/8 -> 256*H/16*W/16
            nn.Conv2d(in_channels=self.out_ch*2, out_channels=self.out_ch*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.out_ch*4),
            nn.ReLU(),
        )
        self.upsample1 = nn.Sequential(     # 128*H/8*W/8 -> 64*H/4*W/4
            nn.ConvTranspose2d(in_channels=self.out_ch*2, out_channels=self.out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.out_ch),
            nn.ReLU()
        )
        self.upsample2 = nn.Sequential(     # 256*H/16*W/16 -> 64*H/4*W/4
            nn.ConvTranspose2d(in_channels=self.out_ch*4, out_channels=self.out_ch, kernel_size=5, stride=4, padding=1, output_padding=1),
            nn.BatchNorm2d(self.out_ch),
            nn.ReLU()
        )


    def forward(self, x):
        x1 = self.downsample1(x)
        x2 = self.downsample2(x1)
        x3 = self.downsample3(x2)

        feature1 = x1
        feature2 = self.upsample1(x2)
        feature3 = self.upsample2(x3)

        cat_feature = torch.cat((feature1, feature2, feature3), dim=1)

        return cat_feature

    @staticmethod
    def check_config(config):
        pass

    @staticmethod
    def build_module(config):
        MultiScaleCNN.check_config(config)
        return MultiScaleCNN()
