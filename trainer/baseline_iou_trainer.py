from genericpath import exists
from tkinter import N
from trainer.trainer_base import TrainerBase
import torch.nn as nn
import torch
import os
import numpy as np
from torchvision import transforms
from utils.loss import CustomLoss, SmoothL1Loss
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from utils.postprocess import non_max_suppression, compute_matches

class BaselineIOUTrainer(TrainerBase):
    def __init__(self, config):
        super(BaselineIOUTrainer, self).__init__()
        self.config = config
        self.max_epoch = config['epoch']
        self.optimizer_config = config['optimizer']
        self.device = torch.device(config['device'])
        self.perception_loss_func = CustomLoss(config['loss_function'])
        self.prediction_loss_func = SmoothL1Loss()
        self.use_error_loss = config['loss_function']['error']
        self.tensorboard_out_path = config['tensorboard_out_path']
        self.data_distributed = config['FP_distribution']
        self.actor_feature_size = config['actor_feature_size']
        self.optimizer = None
        self.data_loader = None

        self.coef_mmd = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda()
        self.coef_mmd.data.fill_(0.005)
        
        pass

    def set_optimizer(self, optimizer_config):
        optimizer_ref = torch.optim.__dict__[self.optimizer_config['type']]
        self.optimizer = optimizer_ref(self.model.parameters(), **optimizer_config['paras'])
    
    def l2_regularization(self, model, l2_alpha):
        l2_loss = []
        for module in model.modules():
            if type(module) is nn.Conv2d:
                l2_loss.append((module.weight ** 2).sum() / 2.0)
        return l2_alpha * sum(l2_loss)

    def run(self):
        if not self.check_ready():
            raise ModuleNotFoundError("The trainer not ready. Plz set model/dataset first")
        torch.autograd.set_detect_anomaly(True)
        self.set_optimizer(self.optimizer_config)
        self.model.set_device(self.device)
        # self.model.set_decode(True)
        self.data_loader = self.dataset.get_data_loader()
        # writer = SummaryWriter(log_dir=self.tensorboard_out_path)

        # Training Loop
        print("device: ", self.device)
        print("Start training!")
        self.global_step = 0
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            for step, data in enumerate(self.data_loader):
                self.step = step
                self.global_step += 1
                self.dataset.load_data_to_gpu(data)
            
                self.model.zero_grad()

                occupancy = data['occupancy'].permute(0, 3, 1, 2)
                occlusion = data['occlusion'].permute(0, 3, 1, 2)
                HDmap = data['HDmap'].permute(0, 3, 1, 2)
                label_map = data['label_map'].permute(0, 3, 1, 2)
                gt_map = data['gt_map'].permute(0, 3, 1, 2)

                ####################
                # Train perception #
                ####################
                input = torch.cat((occupancy, occlusion, HDmap), dim=1)    # ???????????????????????????
                # input = torch.cat((occupancy, occlusion), dim=1)    # ???????????????????????????
                pred, _ = self.model(input)
                perc_loss, cls, loc, corner, cls_loss= self.perception_loss_func(pred, label_map)

                # ??????mmd loss
                mmd_loss = self.model.cauculate_MMD(label_map, pred)
                # ??????L2???????????????
                L2_regularization = self.l2_regularization(self.model, 0.001).cuda()
                # ??????error loss
                gt_dt_error_map = label_map - gt_map
                gt_dt_error_map[:, 0, ...] = label_map[:, 0, ...]
                gt_sim_error_map = pred - gt_map
                error_loss = self.model.cauculate_MMD(gt_dt_error_map, gt_sim_error_map)

                if epoch < 30:   # ????????????????????????????????????cls????????????????????????
                    loss = cls_loss + L2_regularization
                else:
                    if self.use_error_loss:
                        loss = perc_loss + self.coef_mmd * mmd_loss + L2_regularization + 0.001 * error_loss
                    else:
                        loss = perc_loss + self.coef_mmd * mmd_loss + L2_regularization
                loss.backward()
                self.optimizer.step()

                print(
                    f'Epoch: [{epoch + 1:0>{len(str(epoch))}}/{self.max_epoch}]',
                    f'Step: [{step}/{len(self.data_loader)}]',
                    f'Loss-All: {loss.item():.4f}',
                    f'Loss-cls: {cls:.4f}',
                    f'Loss-loc: {loc:.4f}',
                    f'Loss-corner:{corner:.4f}',
                    f'Error-loss: {error_loss.item():.4f}',
                    f'Loss-mmd: {mmd_loss:.4f}',
                    f'L2-regularization: {L2_regularization:.4f}'
                )

            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), \
                            './output/kitti/baseline_pvrcnn_final/' + str(epoch) + ".pt")
