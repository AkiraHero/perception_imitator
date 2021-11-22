from trainer.trainer_base import TrainerBase
import torch.nn as nn
import torch
from torchvision import transforms

class VAEGANTrainerGtbboxGenFpbbox(TrainerBase):
    def __init__(self, config):
        super(VAEGANTrainerGtbboxGenFpbbox, self).__init__()
        self.max_epoch = config['epoch']
        self.optimizer_config = config['optimizer']
        self.device = torch.device(config['device'])
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.data_loader = None
        pass

    def mask(self, fp_hard):
        nonZeroRows = torch.abs(fp_hard).sum(dim=1) > 0

        return nonZeroRows

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
        criterion = nn.BCELoss(size_average=False, reduce=False)  # Initialize BCELoss function
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
                fp_bboxes = data['fp_bboxes_hard']
                cur_batch_size = gt_bboxes.shape[0]

                self.mask(fp_bboxes)

                # self.discriminator_optimizer.zero_grad()
                self.model.discriminator.zero_grad()

                # Get output of target_model
                generator_input = gt_bboxes

                discriminator_input_real = fp_bboxes
                gt_fp_box_label = torch.full((cur_batch_size,),
                                                        real_label, dtype=torch.float, device=self.device)

                # Forward pass real batch through discriminator
                output = self.model.discriminator(discriminator_input_real).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, gt_fp_box_label).mul(self.mask(fp_bboxes)).sum() / cur_batch_size

                # Calculate gradients for D in backward pass
                # errD_real.backward()
                D_x = output.mean().item()

                # Train with all-fake batch
                errD_fake = 0

                # Generate fake image batch with G
                gen_box_fp, mu, logvar = self.model.generator(generator_input)  # 得到每一类的得分

                discriminator_input_fake = gen_box_fp
                generated_label = torch.full((cur_batch_size,),
                                                        fake_label, dtype=torch.float, device=self.device)
                # # Classify all fake batch with D
                output2 = self.model.discriminator(discriminator_input_fake.detach()).view(-1)

                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output2, generated_label).mul(self.mask(fp_bboxes)).sum() / cur_batch_size
                # Calculate the gradients for this batch
                # errD_fake.backward()
                D_G_z1 = output.mean().item()
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
                # netG.zero_grad()
                # self.generator_optimizer.zero_grad()
                self.model.generator.zero_grad()

                # real_fake_label_fullfilled.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output2 = self.model.discriminator(discriminator_input_fake).view(-1)

                # Calculate G's loss based on this output
                errG1 = criterion(output2, gt_fp_box_label).mul(self.mask(fp_bboxes)).sum() / cur_batch_size  # 希望生成的假数据能让D判成1

                KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
                errG_KLD = torch.sum(KLD_element).mul_(-0.5)

                errG = errG1.add_(errG_KLD)
                # errG = errG1
                errG.backward()
                self.logger.log_data("err_G", errG)
                self.logger.log_data("err_G1", errG1)

                D_G_z2 = output.mean().item()

                # Update G
                self.generator_optimizer.step()

                print(
                    f'Epoch: [{epoch + 1:0>{len(str(epoch))}}/{self.max_epoch}]',
                    f'Step: [{step}/{len(self.data_loader)}]',
                    f'Loss-D: {errD.item():.4f}',
                    f'Loss-G: {errG.item():.4f}',
                    f'D(x): {D_x:.4f}',
                    f'D(G(z)): [{D_G_z1:.4f}/{D_G_z2:.4f}]'
                )
            
            if epoch % 10 == 0:
                torch.save(self.model.generator.state_dict(), 'D:/1Pjlab/ADModel_Pro/output/gtbbox_gen_fpbbox_hard_model/' + str(epoch) + ".pt")



