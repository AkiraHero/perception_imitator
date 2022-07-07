from tkinter import N
from trainer.trainer_base import TrainerBase
import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms
from utils.loss import CustomLoss, SmoothL1Loss
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from utils.postprocess import non_max_suppression, compute_matches

class BaselineMMDTrainer(TrainerBase):
    def __init__(self, config):
        super(BaselineMMDTrainer, self).__init__()
        self.config = config
        self.max_epoch = config['epoch']
        self.optimizer_config = config['optimizer']
        self.device = torch.device(config['device'])
        self.perception_loss_func = CustomLoss(config['loss_function'])
        self.prediction_loss_func = SmoothL1Loss()
        self.tensorboard_out_path = config['tensorboard_out_path']
        self.data_distributed = config['FP_distribution']
        self.actor_feature_size = config['actor_feature_size']
        self.optimizer = None
        self.data_loader = None
        
        self.coef_mmd = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda()
        self.coef_mmd.data.fill_(0.0001)
        
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

                ####################
                # Train perception #
                ####################
                input = torch.cat((occupancy, occlusion, HDmap), dim=1)    # 将场景描述共同输入
                pred, _ = self.model(input)
                perc_loss, cls, loc, cls_loss = self.perception_loss_func(pred, label_map)

                mmd_loss = self.model.MMD(label_map, pred)
                L2_regularization = self.l2_regularization(self.model, 0.001).cuda()

                if epoch < 10:   # 为了助于收敛，前三轮只对cls分支进行参数回传
                    loss = cls_loss + L2_regularization
                else:
                    loss = perc_loss + self.coef_mmd * mmd_loss + L2_regularization  # 加入MMD和L2正则化

                loss.backward()
                self.optimizer.step()

                print(
                    f'Epoch: [{epoch + 1:0>{len(str(epoch))}}/{self.max_epoch}]',
                    f'Step: [{step}/{len(self.data_loader)}]',
                    f'Loss-All: {loss.item():.4f}',
                    f'Loss-cls: {cls:.4f}',
                    f'Loss-loc: {loc:.4f}',
                    f'Loss-mmd: {mmd_loss:.4f}',
                    f'L2-regularization: {L2_regularization:.4f}'
                )

            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), \
                            './output/baseline_MMD/' + str(epoch) + ".pt")



