from model.model_base import ModelBase
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, farthest_point_sample
import factory.model_factory as mf
from collections import OrderedDict

class PointNet2Encoder(ModelBase):
    def __init__(self, config):
        super(PointNet2Encoder, self).__init__()
        self.key_pt_sample_num = 2048
        normal_channel = False
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
        self.decoder_names = ['decoder_' + i for i in ['x', 'y', 'z', 'l', 'w', 'h', 'rot', 'cls']]
        self.decoded_dict = OrderedDict()
        for name in self.decoder_names:
            self.add_module(name, mf.ModelFactory.get_model(config['paras']['submodules']['decoder']))

    def get_sample_points(self, data):
        batch_indices = data[:, 0].long()
        keypoints_list = []
        batch_size = batch_indices.unique().shape[0]
        src_points = data[:, 1:4]
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)
            # if self.model_cfg.SAMPLE_METHOD == 'FPS':
            cur_pt_idxs = farthest_point_sample(
                sampled_points[:, :, 0:3].contiguous(), self.key_pt_sample_num
            ).long()

            if sampled_points.shape[1] < self.key_pt_sample_num:
                times = int(self.key_pt_sample_num / sampled_points.shape[1]) + 1
                non_empty = cur_pt_idxs[0, :sampled_points.shape[1]]
                cur_pt_idxs[0] = non_empty.repeat(times)[:self.key_pt_sample_num]
            keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)
            keypoints_list.append(keypoints)
        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3)
        return keypoints

    def forward(self, xyz, gt_boxes):
        pts = xyz[:, :4]
        if self.normal_channel:
            norm = xyz[:, 4:]
        else:
            norm = None
        key_points = self.get_sample_points(pts)
        B, _, _ = key_points.shape
        key_points = key_points.permute([0, 2, 1])

        l1_xyz, l1_points = self.sa1(key_points, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        # encode to mu / sigma
        mu = self.fc4(x)
        log_var = self.fc5(x)
        # get sampled latent code from mu and sigma
        z = self.reparameterize(mu, log_var)

        # reformat gtboxes

        gt_stack = torch.cat(gt_boxes.chunk(gt_boxes.shape[0], dim=0), dim=1).squeeze(0)
        gt_mask = (gt_stack[:, 7] != 0).nonzero()
        assert gt_mask.shape[0] == z.shape[0]
        gt_valid_instance = gt_stack[gt_mask[:, 0], :]
        # decode target is: sample data from pvrcnn
        gt_boxes_dims = {
            'decoder_x': gt_valid_instance[..., 0],
            'decoder_y': gt_valid_instance[..., 1],
            'decoder_z': gt_valid_instance[..., 2],
            'decoder_l': gt_valid_instance[..., 3],
            'decoder_w': gt_valid_instance[..., 4],
            'decoder_h': gt_valid_instance[..., 5],
            'decoder_rot': gt_valid_instance[..., 6],
            'decoder_cls': gt_valid_instance[..., 7]
        }
        for i in self.decoder_names:
            # concatnate scene encoding with gtbox_dim
            decoder_input = torch.cat([z, gt_boxes_dims[i].unsqueeze(1)], dim=1)
            self.decoded_dict[i] = self._modules[i](decoder_input)

        # shape: batch_size * obj_num * [x y z l w h rot cls]
        boxes = torch.cat(list(self.decoded_dict.values()), dim=1)
        return boxes, l3_points, mu, log_var



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
