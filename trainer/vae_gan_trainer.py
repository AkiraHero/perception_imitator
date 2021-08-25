from trainer.trainer_base import TrainerBase
import torch.nn as nn
import torch
from torchvision import transforms
import torch.nn.functional as F

class VAEGANTrainer(TrainerBase):
    def __init__(self, config):
        super(VAEGANTrainer, self).__init__()
        self.max_epoch = config['epoch']
        self.optimizer_config = config['optimizer']
        self.device = torch.device(config['device'])
        self.encoder_optimizer = None
        self.discriminator_optimizer = None
        self.data_loader = None
        pass

    def set_optimizer(self, optimizer_config):
        optimizer_ref = torch.optim.__dict__[self.optimizer_config[0]['type']]
        self.encoder_optimizer = optimizer_ref(self.model.encoder.parameters(), **optimizer_config[0]['paras'])
        optimizer_ref = torch.optim.__dict__[self.optimizer_config[1]['type']]
        self.discriminator_optimizer = optimizer_ref(self.model.encoder.parameters(), **optimizer_config[1]['paras'])

    def run(self):
        if not self.check_ready():
            raise ModuleNotFoundError("The trainer not ready. Plz set model/dataset first")

        self.set_optimizer(self.optimizer_config)
        self.model.set_device(self.device)
        self.data_loader = self.dataset.get_data_loader()
        #todo : when device is known, set_device method should be performed on both dataset and model


        # 初始化Optimizers和损失函数
        criterion = nn.BCELoss()  # Initialize BCELoss function
        # 方便建立真值，Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        # Training Loop
        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        D_x_list = []
        D_z_list = []
        loss_tep1 = 10
        loss_tep2 = 10
        # data transform
        tf_normalize = transforms.Normalize(0.5, 0.5)
        tf_resize = transforms.Resize(14)

        for epoch in range(self.max_epoch):
            for step, data in enumerate(self.data_loader):
                imgs = data[0]
                labels = data[1]
                cur_batch_size = data[0].shape[0]
                self.encoder_optimizer.zero_grad()
                self.discriminator_optimizer.zero_grad()
                # process img
                imgs = tf_normalize(imgs).to(device=self.device)
                generator_input = imgs
                target_score = self.model.target_model(generator_input)

                resized_img_ = tf_resize(tf_normalize(imgs))
                img_shape = resized_img_.shape
                pic_len = img_shape[1] * img_shape[2] * img_shape[3]
                img_ = resized_img_.squeeze().reshape((cur_batch_size, 1, pic_len))
                target_score = target_score.unsqueeze(-2)


                discriminator_input_real = torch.cat((img_, target_score), 2)
                real_fake_label_fullfilled = torch.full((cur_batch_size,),
                                                        real_label, dtype=torch.float, device=self.device)

                # Forward pass real batch through D
                output = self.model.discriminator(discriminator_input_real).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, real_fake_label_fullfilled)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                # Train with all-fake batch
                G_input = imgs
                # Generate fake image batch with G
                G_score, mu, logvar = self.model.encoder(G_input)  # 得到每一类的得分
                # _, G_score = torch.max(G_score.data, 1)

                # 处理压缩图和G生成类别得到可用于输入D的数据

                discriminator_input_fake = torch.cat((img_, G_score), 2)
                real_fake_label_fullfilled.fill_(fake_label)
                # Classify all fake batch with D
                output = self.model.discriminator(discriminator_input_fake.detach()).view(-1)

                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, real_fake_label_fullfilled)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake  # 希望对真实数据接近label1，对于假数据接近label0
                # Update D
                self.discriminator_optimizer.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                # netG.zero_grad()
                real_fake_label_fullfilled.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.model.discriminator(fake).view(-1)
                # Calculate G's loss based on this output
                errG1 = criterion(output, label)  # 希望生成的假数据能让D判成1
                # Calculate gradients for G
                # errG.backward()

                KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
                errG_KLD = F.sigmoid(torch.sum(KLD_element))

                errG = errG1.add_(errG_KLD)
                errG.backward()

                D_G_z2 = output.mean().item()

                # Update G
                self.encoder_optimizer.step()

                # # Output training stats
                # end_time = time.time()
                # run_time = round(end_time - beg_time)
                #
                # print(
                #     f'Epoch: [{epoch + 1:0>{len(str(num_epochs))}}/{num_epochs}]',
                #     f'Step: [{i + 1:0>{len(str(len(dataloader_ori)))}}/{len(dataloader_ori)}]',
                #     f'Loss-D: {errD.item():.4f}',
                #     f'Loss-G: {errG.item():.4f}',
                #     f'D(x): {D_x:.4f}',
                #     f'D(G(z)): [{D_G_z1:.4f}/{D_G_z2:.4f}]',
                #     f'Time: {run_time}s',
                #     end='\r'
                # )
                # # Save Losses for plotting later
                # G_losses.append(errG.item())
                # D_losses.append(errD.item())
                #
                # # Save D(X) and D(G(z)) for plotting later
                # D_x_list.append(D_x)
                # D_z_list.append(D_G_z2)
                #
                # # Save the Best Model
                # if errG < loss_tep1 and epoch > 10:
                #     torch.save(netG.state_dict(), './results/VAE_Mnist2/model_errG.pt')
                #     loss_tep1 = errG
                # if epoch % 10 == 0:
                #     torch.save(netG.state_dict(), './results/VAE_Mnist2/model_%d.pt' % (epoch))



