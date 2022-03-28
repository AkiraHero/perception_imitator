from model.model_base import ModelBase
import factory.model_factory as ModelFactory
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, geom, ratio):
        super(Decoder, self).__init__()
        self.geometry = geom
        self.ratio = ratio

        self.target_mean = [0.008, 0.001, 0.202, 0.2, 0.43, 1.368]
        self.target_std_dev = [0.866, 0.5, 0.954, 0.668, 0.09, 0.111]

    def forward(self, x):
        '''
        :param x: Tensor 6-channel geometry 
        6 channel map of [cos(yaw), sin(yaw), dx, dy, log(w), log(l)]
        Shape of x: (B, C=6, H=352, W=400)
        or  Tensor 11-channel geometry 
        6 channel map of [cos(yaw), sin(yaw), dx, dy, log(w), log(l), yaw_logvar, x_logvar, y_logvar, w_logvar, l_logvar]
        Shape of x: (B, C=11, H=352, W=400)
        :return: Concatenated Tensor of 8 channel geometry map of bounding box corners
        8 channel are [rear_left_x, rear_left_y,
                        rear_right_x, rear_right_y,
                        front_right_x, front_right_y,
                        front_left_x, front_left_y]
        Return tensor has a shape (B, C=8, H=352, W=400), and is located on the same device as x
        '''
        # Tensor in (B, C, H, W)
        device = torch.device('cpu')
        if x.is_cuda:
            device = x.get_device()

        # for i in range(6):
        #     x[:, i, :, :] = x[:, i, :, :] * self.target_std_dev[i] + self.target_mean[i]
        
        if x.shape[1] == 6:
            cos_t, sin_t, dx, dy, log_w, log_l = torch.chunk(x, 6, dim=1)
            theta = torch.atan2(sin_t, cos_t)
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)

            x = torch.arange(self.geometry[3], self.geometry[2], -self.ratio, dtype=torch.float32, device=device)
            y = torch.arange(self.geometry[1], self.geometry[0], -self.ratio, dtype=torch.float32, device=device)
            xx, yy = torch.meshgrid([x, y])
            centre_y = yy + dy
            centre_x = xx + dx
            l = log_l.exp()
            w = log_w.exp()
            rear_left_x = centre_x - l/2 * cos_t - w/2 * sin_t
            rear_left_y = centre_y - l/2 * sin_t + w/2 * cos_t
            rear_right_x = centre_x - l/2 * cos_t + w/2 * sin_t
            rear_right_y = centre_y - l/2 * sin_t - w/2 * cos_t
            front_right_x = centre_x + l/2 * cos_t + w/2 * sin_t
            front_right_y = centre_y + l/2 * sin_t - w/2 * cos_t
            front_left_x = centre_x + l/2 * cos_t - w/2 * sin_t
            front_left_y = centre_y + l/2 * sin_t + w/2 * cos_t

            decoded_reg = torch.cat([rear_left_x, rear_left_y, rear_right_x, rear_right_y,
                                    front_right_x, front_right_y, front_left_x, front_left_y], dim=1)
        elif x.shape[1] == 11:
            cos_t, sin_t, dx, dy, log_w, log_l, yaw_logvar, x_logvar, y_logvar, w_logvar, l_logvar = torch.chunk(x, 11, dim=1)
            theta = torch.normal(torch.atan2(sin_t, cos_t), yaw_logvar.mul(0.5).exp_())    # 使用torhc.normal进行正态分布采样
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)

            x = torch.arange(self.geometry[3], self.geometry[2], -self.ratio, dtype=torch.float32, device=device)
            y = torch.arange(self.geometry[1], self.geometry[0], -self.ratio, dtype=torch.float32, device=device)
            xx, yy = torch.meshgrid([x, y])
            centre_y = torch.normal((yy + dy), y_logvar.mul(0.5).exp_())
            centre_x = torch.normal((xx + dx), x_logvar.mul(0.5).exp_())
            l = torch.normal(log_l.exp(), l_logvar.mul(0.5).exp_())
            w = torch.normal(log_w.exp(), w_logvar.mul(0.5).exp_())
            rear_left_x = centre_x - l/2 * cos_t - w/2 * sin_t
            rear_left_y = centre_y - l/2 * sin_t + w/2 * cos_t
            rear_right_x = centre_x - l/2 * cos_t + w/2 * sin_t
            rear_right_y = centre_y - l/2 * sin_t - w/2 * cos_t
            front_right_x = centre_x + l/2 * cos_t + w/2 * sin_t
            front_right_y = centre_y + l/2 * sin_t - w/2 * cos_t
            front_left_x = centre_x + l/2 * cos_t - w/2 * sin_t
            front_left_y = centre_y + l/2 * sin_t + w/2 * cos_t

            decoded_reg = torch.cat([rear_left_x, rear_left_y, rear_right_x, rear_right_y,
                                    front_right_x, front_right_y, front_left_x, front_left_y], dim=1)

        return decoded_reg

class MultiScaleCNNCls(ModelBase):
    def __init__(self, config):
        super(MultiScaleCNNCls, self).__init__()
        self.backbone = ModelFactory.ModelFactory.get_model(config['paras']['submodules']['backbone'])
        self.head = ModelFactory.ModelFactory.get_model(config['paras']['submodules']['head'])
        
        self.geom = [-40, 40, 0.0, 70.4]
        self.ratio = 0.2
        self.corner_decoder = Decoder(self.geom, self.ratio)
        self.use_decode = False

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

    def set_decode(self, decode):
        self.use_decode = decode

    def forward(self, x):
        features = self.backbone(x)
        cls, reg = self.head(features)

        if self.use_decode:
            decoded = self.corner_decoder(reg)
            pred = torch.cat([cls, reg, decoded], dim=1)
        else:
            pred = torch.cat([cls, reg], dim=1)

        return pred