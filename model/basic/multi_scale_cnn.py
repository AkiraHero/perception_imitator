from sklearn.preprocessing import scale
from model.model_base import ModelBase
import factory.model_factory as ModelFactory
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from .utils import *

class MultiScaleCNN(ModelBase):
    def __init__(self, config):
        super(MultiScaleCNN, self).__init__()
        self.in_ch = config['paras']['input_channel']
        self.out_ch = config['paras']['one_feature_channel']
        self.add_attention = config['paras']['add_attention']

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
        
        if self.add_attention:
            self.att_Unet = AttU_Net(n_channels=self.in_ch)   

    def forward(self, x):
        x1 = self.downsample1(x)
        x2 = self.downsample2(x1)
        x3 = self.downsample3(x2)

        if self.add_attention:
            att_mask = self.att_Unet(x)
            att_mask_x1 = nn.functional.interpolate(att_mask, scale_factor=0.25, mode='bilinear')
            att_mask_x2 = nn.functional.interpolate(att_mask_x1, scale_factor=0.5, mode='bilinear')
            att_mask_x3 = nn.functional.interpolate(att_mask_x2, scale_factor=0.5, mode='bilinear')

            # elementwise mul
            mask_x1 = x1 * att_mask_x1
            mask_x2 = x2 * att_mask_x2
            mask_x3 = x3 * att_mask_x3

            # resblock
            x1 = x1 + mask_x1
            x2 = x2 + mask_x2
            x3 = x3 + mask_x3

        feature1 = x1
        feature2 = self.upsample1(x2)
        feature3 = self.upsample2(x3)

        cat_feature = torch.cat((feature1, feature2, feature3), dim=1)

        if self.add_attention:
            return cat_feature, att_mask
        else:
            return cat_feature

    @staticmethod
    def check_config(config):
        pass

    @staticmethod
    def build_module(config):
        MultiScaleCNN.check_config(config)
        return MultiScaleCNN()

######################
# Attention Unet Model
######################
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class AttU_Net(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, scale_factor=1):
        super(AttU_Net, self).__init__()
        filters = np.array([64, 128, 256, 512, 1024])
        filters = filters // scale_factor
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.scale_factor = scale_factor
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=n_channels, ch_out=filters[0])
        self.Conv2 = conv_block(ch_in=filters[0], ch_out=filters[1])
        self.Conv3 = conv_block(ch_in=filters[1], ch_out=filters[2])
        self.Conv4 = conv_block(ch_in=filters[2], ch_out=filters[3])
        self.Conv5 = conv_block(ch_in=filters[3], ch_out=filters[4])

        self.Up5 = up_conv(ch_in=filters[4], ch_out=filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(ch_in=filters[4], ch_out=filters[3])

        self.Up4 = up_conv(ch_in=filters[3], ch_out=filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(ch_in=filters[3], ch_out=filters[2])

        self.Up3 = up_conv(ch_in=filters[2], ch_out=filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(ch_in=filters[2], ch_out=filters[1])

        self.Up2 = up_conv(ch_in=filters[1], ch_out=filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=filters[0] // 2)
        self.Up_conv2 = conv_block(ch_in=filters[1], ch_out=filters[0])

        self.Conv_1x1 = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)
        self.Sigmoid = nn.Sigmoid()
        self.GumbelSoftmax = CNNGumbelSoftmax()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.Up3(x3)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.Sigmoid(d1)
        d1 = self.GumbelSoftmax(d1, hard=True)

        return d1