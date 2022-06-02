from tkinter import N
from trainer.trainer_base import TrainerBase
import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms
from utils.loss import SmoothL1Loss
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from utils.postprocess import non_max_suppression, compute_matches

class ActornoiseTrainer(TrainerBase):
    def __init__(self, config):
        super(ActornoiseTrainer, self).__init__()
        self.max_epoch = config['epoch']
        self.optimizer_config = config['optimizer']
        self.device = torch.device(config['device'])
        self.tensorboard_out_path = config['tensorboard_out_path']
        self.optimizer = None
        self.data_loader = None

    def set_optimizer(self, optimizer_config):
        optimizer_ref = torch.optim.__dict__[self.optimizer_config['type']]
        self.optimizer = optimizer_ref(self.model.parameters(), **optimizer_config['paras'])

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

                self.cls_criterion = nn.BCEWithLogitsLoss().to(self.device)
                self.reg_criterion = SmoothL1Loss().to(self.device)
            
                input = data['GT_bbox']
                label = data['label']
                cls_label = label[:, 0]
                box_label = label[:, 1:6]
                waypoint_label = label[:, 6:]

                self.model.zero_grad()
                cls, box, waypoint = self.model(input)

                cls_loss = self.cls_criterion(cls.squeeze(), cls_label)
                box_loss = self.reg_criterion(box[cls_label.bool()], box_label[cls_label.bool()])
                waypoint_loss =self.reg_criterion(waypoint[cls_label.bool()], waypoint_label[cls_label.bool()])

                loss = cls_loss + box_loss + waypoint_loss
                loss.backward()
                self.optimizer.step()

                # writer.add_scalar("loss_all", loss, self.global_step)
                # writer.add_scalar("cls", cls_loss, self.global_step)
                # writer.add_scalar("reg", reg_loss, self.global_step)

            print(
                    f'Epoch: [{epoch + 1:0>{len(str(epoch))}}/{self.max_epoch}]',
                    # f'Step: [{step}/{len(self.data_loader)}]',
                    f'Loss-All: {loss:.4f}',
                    f'cls: {cls_loss:.4f}',
                    f'box: {box_loss:.4f}',
                    f'waypoint: {waypoint_loss:.4f}',
                )

            if epoch % 5 == 0:
                torch.save(self.model.state_dict(), \
                            './output/actor_noise/' + str(epoch) + ".pt")



