from trainer.trainer_base import TrainerBase
import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms

from utils.loss import  GaussianMixtureFunctions

class GMVaeGanTrainer(TrainerBase):
    def __init__(self, config):
        super(GMVaeGanTrainer, self).__init__()
        self.max_epoch = config['epoch']
        self.optimizer_config = config['optimizer']
        self.device = torch.device(config['device'])
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.data_loader = None
        self.gumnble_temp = config['gumble_temp']
        self.hard_gumble = config['hard_gumble']
        self.w_rec = config['rec_weight']
        self.w_gauss = config['gauss_weight']
        self.w_categ = config['categ_weight']
        self.rec_type = config['recon_type']

    def set_optimizer(self, optimizer_config):
        optimizer_ref = torch.optim.__dict__[self.optimizer_config[0]['type']]
        self.generator_optimizer = optimizer_ref(self.model.generator.parameters(), **optimizer_config[0]['paras'])
        optimizer_ref = torch.optim.__dict__[self.optimizer_config[1]['type']]
        self.discriminator_optimizer = optimizer_ref(self.model.discriminator.parameters(), **optimizer_config[1]['paras'])

    def run(self):
        if not self.check_ready():
            raise ModuleNotFoundError("The trainer not ready. Plz set model/dataset first")
        torch.autograd.set_detect_anomaly(True)
        self.set_optimizer(self.optimizer_config)
        self.model.set_device(self.device)
        self.data_loader = self.dataset.get_data_loader()

        # 初始化Optimizers和损失函数
        discriminator_criterion = nn.BCELoss(size_average=False, reduce=False)  # Initialize BCELoss function for discriminator
        GM_creterion = GaussianMixtureFunctions()                               # Initialize Loss function for Gassian Mixture Model

        # 方便建立真值，Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        # Training Loop
        # Lists to keep track of progress
        self.global_step = 0

        for epoch in range(self.max_epoch):
            self.epoch = epoch
            for step, data in enumerate(self.data_loader):
                self.step = step
                self.global_step += 1
                self.dataset.load_data2gpu(data)

                gt_bboxes = data['gt_bboxes']
                fp_bboxes = data['fp_bboxes_all']
                cur_batch_size = gt_bboxes.shape[0]
                
                ############################
                # (1) Update D network
                ###########################
                self.discriminator_optimizer.zero_grad()

                # Get output of target_model
                generator_input = gt_bboxes # bs*640
                discriminator_input_real = torch.cat((generator_input, fp_bboxes), 1)   # bs * (640+140)

                gt_fp_box_label = torch.full((discriminator_input_real.shape[0],),
                                                        real_label, dtype=torch.float, device=self.device)                                  

                # Forward pass real batch through discriminator
                output = self.model.discriminator(discriminator_input_real).view(-1)
                # output = normalize_gradient(self.model.discriminator, discriminator_input_real).view(-1)

                # Calculate loss on all-real batch
                errD_real = discriminator_criterion(output, gt_fp_box_label).sum() / cur_batch_size

                # Calculate gradients for D in backward pass
                D_x = output.mean().item()

                # Train with all-fake batch
                # Generate fake image batch with G
                GMmodel_out = self.model.generator(generator_input)
                z, gen_fp_bboxes = GMmodel_out['gaussian'], GMmodel_out['x_rec'] 
                logits, prob_cat = GMmodel_out['logits'], GMmodel_out['prob_cat']
                y_mu, y_var = GMmodel_out['y_mean'], GMmodel_out['y_var']
                mu, var = GMmodel_out['mean'], GMmodel_out['var']

                discriminator_input_fake = torch.cat((generator_input, gen_fp_bboxes), 1)
                generated_label = torch.full((discriminator_input_fake.shape[0],),
                                                        fake_label, dtype=torch.float, device=self.device)

                # # Classify all fake batch with D
                output2 = self.model.discriminator(discriminator_input_fake.detach()).view(-1)

                # Calculate D's loss on the all-fake batch
                errD_fake = discriminator_criterion(output2, generated_label).sum() / cur_batch_size

                # Calculate the gradients for this batch
                D_G_z1 = output2.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake  # 希望对真实数据接近label1，对于假数据接近label0

                # # Update D
                errD.backward()
                self.logger.log_data("errD", errD)
                self.logger.log_data("err_real", errD_real)
                self.logger.log_data("err_fake", errD_fake)
                self.discriminator_optimizer.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.generator_optimizer.zero_grad()

                # Since we just updated D, perform another forward pass of all-fake batch through D
                output2 = self.model.discriminator(discriminator_input_fake).view(-1)

                # Calculate G's loss based on this output
                errG1 = discriminator_criterion(output2, gt_fp_box_label).sum() / cur_batch_size  # 希望生成的假数据能让D判成1

                # Calculate Gaussian Mixture Loss
                loss_rec = GM_creterion.reconstruction_loss(fp_bboxes, gen_fp_bboxes, self.rec_type) # reconstruction loss
                loss_gauss = GM_creterion.gaussian_loss(z, mu, var, y_mu, y_var) # gaussian loss
                loss_cat = - GM_creterion.entropy(logits, prob_cat) - np.log(0.1) # categorical loss
                
                errG_GM = self.w_rec * loss_rec + self.w_gauss * loss_gauss + self.w_categ * loss_cat

                errG = errG1.add(errG_GM)
                errG.backward()
                self.logger.log_data("err_G", errG)
                self.logger.log_data("err_G1", errG1)

                D_G_z2 = output2.mean().item()

                # Update G
                self.generator_optimizer.step()
                
                print(
                    f'Epoch: [{epoch + 1:0>{len(str(epoch))}}/{self.max_epoch}]',
                    f'Step: [{step}/{len(self.data_loader)}]',
                    f'Loss-D: {errD.item():.4f}',
                    f'Loss-G: {errG.item():.4f}',
                    f'Loss-rec: {loss_rec.item():.4f}',
                    f'Loss-gauss: {loss_gauss.item():.4f}',
                    f'Loss-categ: {loss_cat.item():.4f}',
                    f'D(x): {D_x:.4f}',
                    f'D(G(z)): [{D_G_z1:.4f}/{D_G_z2:.4f}]'
                )

            if epoch % 10 == 0:
                torch.save(self.model.generator.state_dict(), './output/GM_gen_fp/' + str(epoch) + ".pt")
