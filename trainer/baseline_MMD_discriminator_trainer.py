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

class BaselineMMDDiscrimTrainer(TrainerBase):
    def __init__(self, config):
        super(BaselineMMDDiscrimTrainer, self).__init__()
        self.config = config
        self.max_epoch = config['epoch']
        self.optimizer_config = config['optimizer']
        self.device = torch.device(config['device'])
        self.perception_loss_func = CustomLoss(config['loss_function'])
        self.prediction_loss_func = SmoothL1Loss()
        self.tensorboard_out_path = config['tensorboard_out_path']
        self.data_distributed = config['FP_distribution']
        self.actor_feature_size = config['actor_feature_size']
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.data_loader = None
        
        self.coef_mmd = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda()
        self.coef_mmd.data.fill_(0.0001)

        self.coef_dis = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda()
        self.coef_dis.data.fill_(0.01)
        
        pass

    def set_optimizer(self, optimizer_config):
        optimizer_ref = torch.optim.__dict__[self.optimizer_config[0]['type']]
        self.generator_optimizer = optimizer_ref(self.model.generator.parameters(), **optimizer_config[0]['paras'])
        optimizer_ref = torch.optim.__dict__[self.optimizer_config[1]['type']]
        self.discriminator_optimizer = optimizer_ref(self.model.discriminator.parameters(), **optimizer_config[1]['paras'])

    def l2_regularization(self, model, l2_alpha):
        l2_loss = []
        for module in model.modules():
            if type(module) is nn.Conv2d:
                l2_loss.append((module.weight ** 2).sum() / 2.0)
        return l2_alpha * sum(l2_loss)

    def get_discriminator_input(self, real, target):
        bs = real.shape[0]
        mask = (real[:,0,...] > 0)

        all_input = []
        for i in range(bs):
            batch_input = target[i, ..., mask[i]]
            if batch_input.shape[-1] <= 256:
                extend = torch.from_numpy(np.zeros((batch_input.shape[0], 256-batch_input.shape[-1]))).to(torch.float32).cuda()
                batch_input = torch.cat((batch_input, extend), 1)
            else:
                batch_input = batch_input.narrow(1, 0, 256)

            all_input.append(batch_input)
        
        all_input = torch.cat(all_input, dim=0)
        return all_input

    def run(self):
        if not self.check_ready():
            raise ModuleNotFoundError("The trainer not ready. Plz set model/dataset first")
        torch.autograd.set_detect_anomaly(True)
        self.set_optimizer(self.optimizer_config)
        self.model.set_device(self.device)
        self.data_loader = self.dataset.get_data_loader()

        # 初始化Optimizers和损失函数
        criterion = nn.BCELoss(size_average=False, reduce=True)  # Initialize BCELoss function
        # 方便建立真值，Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

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
            
                self.generator_optimizer.zero_grad()

                occupancy = data['occupancy'].permute(0, 3, 1, 2)
                occlusion = data['occlusion'].permute(0, 3, 1, 2)
                HDmap = data['HDmap'].permute(0, 3, 1, 2)
                label_map = data['label_map'].permute(0, 3, 1, 2)

                ####################
                # Train perception #
                ####################
                input = torch.cat((occupancy, occlusion, HDmap), dim=1)    # 将场景描述共同输入
                pred, _ = self.model.generator(input)
                perc_loss, cls, loc, cls_loss = self.perception_loss_func(pred, label_map)

                mmd_loss = self.model.generator.cauculate_MMD(label_map, pred)
                L2_regularization = self.l2_regularization(self.model.generator, 0.001).cuda()

                if epoch < 10:   # 为了助于收敛，前三轮只对cls分支进行参数回传
                    loss = cls_loss + L2_regularization
                    loss.backward()
                    self.generator_optimizer.step()
                    errD = torch.tensor([0])
                    errG = torch.tensor([0])
                elif epoch < 200:
                    loss = perc_loss + self.coef_mmd * mmd_loss + L2_regularization  # 加入MMD和L2正则化
                    loss.backward()
                    self.generator_optimizer.step()
                    errD = torch.tensor([0])
                    errG = torch.tensor([0])
                else:
                    ### Update Discriminator ###
                    self.discriminator_optimizer.zero_grad()

                    discriminator_input_real = self.get_discriminator_input(label_map, label_map)
                    cur_batch_size = discriminator_input_real.shape[0]

                    true_label = torch.full((cur_batch_size,),
                                                        real_label, dtype=torch.float, device=self.device)

                    # Forward pass real batch through discriminator
                    output = self.model.discriminator(discriminator_input_real).view(-1)
                    # Calculate loss on all-real batch
                    errD_real = criterion(output, true_label) / cur_batch_size
                    # Calculate gradients for D in backward pass
                    D_x = output.mean().item()

                    # Train with all-fake batch
                    errD_fake = 0

                    discriminator_input_fake = self.get_discriminator_input(label_map, pred)
                    generated_label = torch.full((cur_batch_size,),
                                                            fake_label, dtype=torch.float, device=self.device)
                    # # Classify all fake batch with D
                    output2 = self.model.discriminator(discriminator_input_fake.detach()).view(-1)

                    # Calculate D's loss on the all-fake batch
                    errD_fake = criterion(output2, generated_label) / cur_batch_size
                    # Calculate the gradients for this batch
                    D_G_z1 = output.mean().item()
                    # Add the gradients from the all-real and all-fake batches
                    errD = errD_real + errD_fake  # 希望对真实数据接近label1，对于假数据接近label0
                    # Update D
                    errD.backward()
                    self.discriminator_optimizer.step()


                    ### Update Generator(Baseline) ###
                    self.generator_optimizer.zero_grad()
                    # real_fake_label_fullfilled.fill_(real_label)  # fake labels are real for generator cost
                    # Since we just updated D, perform another forward pass of all-fake batch through D
                    output2 = self.model.discriminator(discriminator_input_fake).view(-1)

                    # output = self.model.discriminator(discriminator_input_fake).view(-1)
                    # Calculate G's loss based on this output
                    errG = criterion(output2, true_label) / cur_batch_size  # 希望生成的假数据能让D判成1
                    loss = perc_loss + self.coef_mmd * mmd_loss + L2_regularization + self.coef_dis * errG  # 加入MMD和L2正则化

                    loss.backward()
                    self.generator_optimizer.step()

                print(
                    f'Epoch: [{epoch + 1:0>{len(str(epoch))}}/{self.max_epoch}]',
                    f'Step: [{step}/{len(self.data_loader)}]',
                    f'Loss-All: {loss.item():.4f}',
                    f'Loss-cls: {cls:.4f}',
                    f'Loss-loc: {loc:.4f}',
                    f'Loss-mmd: {mmd_loss:.4f}',
                    f'L2-regularization: {L2_regularization:.4f}',
                    f'Loss-D:: {errD.item():.4f}',
                    f'Loss-G: {errG.item():.4f}'
                )

            if epoch % 10 == 0:
                torch.save(self.model.generator.state_dict(), \
                            './output/baseline_MMD_discrim_pvrcnn/' + str(epoch) + ".pt")



