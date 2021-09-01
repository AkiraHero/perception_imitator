from model.model_base import ModelBase
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


class PointNet2Encoder(ModelBase):
    def __init__(self, config):
        super(PointNet2Encoder, self).__init__()
        normal_channel = True
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 64)

        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(64, 32)


    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        # encode to mu / sigma
        mu = self.fc4(x)
        log_var = self.fc5(x)

        # decode
        z = self.reparameterize(mu, log_var)


        return x, l3_points



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
        PointNet2Encoder.check_config(config)
        return PointNet2Encoder()
