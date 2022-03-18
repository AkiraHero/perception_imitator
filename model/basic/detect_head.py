from model.model_base import ModelBase
import factory.model_factory as ModelFactory
import torch.nn as nn
import torch.nn.functional as F
import torch

class DetectHead(ModelBase):
    def __init__(self, config):
        super(DetectHead, self).__init__()
        self._input_size = config['paras']['input_size']
        self._mid_layer_num = config['paras']['mid_layer_num']

        self.upsample = nn.Sequential(     # 192*H/4*W/4 -> 64*H*W
            nn.ConvTranspose2d(in_channels=self._input_size[0], out_channels=self._mid_layer_num, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self._mid_layer_num),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self._mid_layer_num, out_channels=self._mid_layer_num, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self._mid_layer_num),
            nn.ReLU()
        )

        self.clshead = nn.Conv2d(in_channels=self._mid_layer_num, out_channels=1, kernel_size=1)
        self.reghead = nn.Conv2d(in_channels=self._mid_layer_num, out_channels=6, kernel_size=1)

    def forward(self, x):
        x = self.upsample(x)

        cls = torch.sigmoid(self.clshead(x))
        reg = self.reghead(x)

        return cls, reg

    @staticmethod
    def check_config(config):
        pass

    @staticmethod
    def build_module(config):
        DetectHead.check_config(config)
        return DetectHead()
