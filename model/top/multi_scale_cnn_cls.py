from turtle import pos
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

class MMD(nn.Module):
    def __init__(self):
        super(MMD, self).__init__()

    def compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        # expand does not allocate new memory
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)
      
        return torch.exp(-kernel_input) # (x_size, y_size)

    def forward(self, real, gen):
        bs = real.shape[0]
        mask = (real[:,0,...] > 0)

        all_x = []
        all_y = []
        mmd = 0.0
        for i in range(bs):
            b_x = real[i, ..., mask[i]]
            b_y = gen[i, ..., mask[i]]

            all_x.append(b_x)
            all_y.append(b_y)

        all_x = torch.cat(all_x, dim=-1)
        all_y = torch.cat(all_y, dim=1)
        x_mean, x_std = all_x.mean(axis=1), all_x.std(axis=1)
        y_mean, y_std = all_y.mean(axis=1), all_y.std(axis=1)

        for layer in range(all_x.shape[0] - 1):   # 针对除了cls外的每一层参数分别计算mmd，最后相加
            X = (all_x[layer + 1].unsqueeze(0) - x_mean[layer + 1]) / x_std[layer + 1]
            Y = (all_y[layer + 1].unsqueeze(0) - y_mean[layer + 1]) / y_std[layer + 1]

            if x_std[layer + 1] == 0:
                X = all_x[layer + 1].unsqueeze(0)
            if y_std[layer + 1] == 0:
                Y = all_y[layer + 1].unsqueeze(0)

            x_kernal = self.compute_kernel(X, X)
            y_kernal = self.compute_kernel(Y, Y)
            xy_kernal = self.compute_kernel(X, Y)

            layer_mmd = y_kernal.mean() - 2 * xy_kernal.mean() + x_kernal.mean()
            if math.isnan(layer_mmd):
                continue
            mmd += layer_mmd
        return mmd

class MultiScaleCNNCls(ModelBase):
    def __init__(self, config):
        super(MultiScaleCNNCls, self).__init__()
        self.backbone = ModelFactory.ModelFactory.get_model(config['paras']['submodules']['backbone'])
        self.head = ModelFactory.ModelFactory.get_model(config['paras']['submodules']['head'])
        self.cauculate_MMD = MMD()
        if config['paras']['submodules']['prediction']:
            self.prediction = ModelFactory.ModelFactory.get_model(config['paras']['submodules']['prediction'])
        self.pos_encode = config['paras']['submodules']['position_encoding']
        
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

    def positionalencoding2d(self, d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        # d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        # pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        # pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        # pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        # pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[0:d_model:2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[1:d_model:2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe

    def forward(self, x):
        if self.pos_encode:
            # upnear = nn.UpsamplingNearest2d(scale_factor=16)    # 15*[22,25] = [352,400]
            pos_encoding = self.positionalencoding2d(64,352,400).unsqueeze(0).repeat(x.shape[0], 1, 1, 1).cuda()
            # pos_encoding = upnear(pos_encoding)

            x = torch.cat((x, pos_encoding), dim=1)

        if torch.is_tensor(self.backbone(x)): 
            features = self.backbone(x)
        elif isinstance(self.backbone(x), tuple):
            features, attention_mask = self.backbone(x)
        else:
            raise Exception("dim error")
        cls, reg = self.head(features)

        if self.use_decode:
            decoded = self.corner_decoder(reg)
            pred = torch.cat([cls, reg, decoded], dim=1)
        else:
            pred = torch.cat([cls, reg], dim=1)

        if 'attention_mask' in locals().keys():
            return pred, features, attention_mask
        else:
            return pred, features