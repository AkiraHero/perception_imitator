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
        self.ratio = 1/4  # 对原图进行的缩放比例，节省MMD计算量
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=4, padding=1),
            nn.MaxPool2d(2),
            nn.Sigmoid(),
        )
        self.img_norm = nn.Sigmoid()

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        '''
        将源域数据和目标域数据转化为核矩阵，即上文中的K
        Params: 
            source: 源域数据（n * len(x))
            target: 目标域数据（m * len(y))
            kernel_mul: 
            kernel_num: 取不同高斯核的数量
            fix_sigma: 不同高斯核的sigma值
        Return:
            sum(kernel_val): 多个核矩阵之和
        '''
        n_samples = int(source.size()[0])+int(target.size()[0])# 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
        total = torch.cat([source, target], dim=0)#将source,target按列方向合并
        #将total复制（n+m）份
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        #求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
        L2_distance = ((total0-total1)**2).sum(2) 
        #调整高斯核函数的sigma值
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        #高斯核函数的数学表达式
        kernel_val = [torch.exp(-L2_distance / (bandwidth_temp + 1e-6)) for bandwidth_temp in bandwidth_list]
        #得到最终的核矩阵
        return sum(kernel_val)#/len(kernel_val)

    def mmd_rbf(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        '''
        计算源域数据和目标域数据的MMD距离
        Params: 
            source: 源域数据（n * len(x))
            target: 目标域数据（m * len(y))
            kernel_mul: 
            kernel_num: 取不同高斯核的数量
            fix_sigma: 不同高斯核的sigma值
        Return:
            loss: MMD loss
        '''
        batch_size = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
        kernels = self.guassian_kernel(source, target,
            kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        #根据式（3）将核矩阵分成4部分
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss #因为一般都是n==m，所以L矩阵一般不加入计算

    def get_features(self, images):
        out_feats = []

        # 对特征进行MMD
        feat = self.conv(images)
        out_feats.append(feat.contiguous().view(feat.size(0), -1))

        # 对原始图缩放标准化后进行MMD
        simp_img = nn.functional.interpolate(images, scale_factor=self.ratio, mode='bilinear', align_corners=False)
        # simp_img = self.img_norm(simp_img)
        out_feats.append(simp_img.contiguous().view(simp_img.size(0), -1))

        return out_feats

    @staticmethod
    def compute_kernel(x, y):
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
        x = self.get_features(real)
        y = self.get_features(gen)

        mmd2 = 0.0
        for i in range(len(x)):
            mmd2 += self.mmd_rbf(x[i], y[i])

        return mmd2

class MultiScaleCNNCls(ModelBase):
    def __init__(self, config):
        super(MultiScaleCNNCls, self).__init__()
        self.backbone = ModelFactory.ModelFactory.get_model(config['paras']['submodules']['backbone'])
        self.head = ModelFactory.ModelFactory.get_model(config['paras']['submodules']['head'])
        if config['paras']['submodules']['prediction'] != "None":
            self.prediction = ModelFactory.ModelFactory.get_model(config['paras']['submodules']['prediction'])
        if config['paras']['submodules']['MMD'] != "None":
            self.MMD = MMD()
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
            pos_encoding = self.positionalencoding2d(128, 352, 400).unsqueeze(0).repeat(x.shape[0], 1, 1, 1).cuda()
            x = torch.concat((x, pos_encoding), dim=1)

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