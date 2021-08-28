from trainer.trainer_base import TrainerBase
import torch.nn as nn
import torch
from torchvision import transforms


class VAEGANTrainer(TrainerBase):
    def __init__(self, config):
        super(VAEGANTrainer, self).__init__()
        self.max_epoch = config['epoch']
        self.optimizer_config = config['optimizer']
        self.device = torch.device(config['device'])
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.data_loader = None
        pass

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
        criterion = nn.BCELoss()  # Initialize BCELoss function
        # 方便建立真值，Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        # Training Loop
        # Lists to keep track of progress

        # data transform
        tf_normalize = transforms.Normalize(0.5, 0.5)
        tf_resize = transforms.Resize(14)

        for epoch in range(self.max_epoch):
            for step, data in enumerate(self.data_loader):
                imgs = data[0].to(device=self.device)
                # gt_labels = data[1]
                cur_batch_size = data[0].shape[0]

                # self.discriminator_optimizer.zero_grad()
                self.model.discriminator.zero_grad()
                # process img
                resized_img_ = tf_resize(imgs)
                img_normalized = tf_normalize(resized_img_)
                img_shape = img_normalized.shape
                pic_len = img_shape[1] * img_shape[2] * img_shape[3]
                flattened_img_vector = img_normalized.squeeze().reshape((cur_batch_size, 1, pic_len))

                # Get output of target_model
                generator_input = imgs
                target_score = self.model.target_model(generator_input)

                # Get discriminator input: flattened image and target model output
                target_score = target_score.unsqueeze(-2)
                discriminator_input_real = torch.cat((flattened_img_vector, target_score), 2)
                target_model_label = torch.full((cur_batch_size,),
                                                        real_label, dtype=torch.float, device=self.device)

                # Forward pass real batch through discriminator
                output = self.model.discriminator(discriminator_input_real).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, target_model_label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                # Train with all-fake batch
                errD_fake = 0

                # generator_input = imgs
                # Generate fake image batch with G
                G_score, mu, logvar = self.model.generator(generator_input)  # 得到每一类的得分
                # _, G_score = torch.max(G_score.data, 1)
                G_score = G_score.unsqueeze(-2)
                # # 处理压缩图和G生成类别得到可用于输入D的数据

                discriminator_input_fake = torch.cat((flattened_img_vector, G_score), 2)
                generated_label = torch.full((cur_batch_size,),
                                                        fake_label, dtype=torch.float, device=self.device)
                # # Classify all fake batch with D
                output2 = self.model.discriminator(discriminator_input_fake.detach()).view(-1)

                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output2, generated_label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake  # 希望对真实数据接近label1，对于假数据接近label0
                # # Update D
                # errD.backward()
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

                # output = self.model.discriminator(discriminator_input_fake).view(-1)
                # Calculate G's loss based on this output
                errG1 = criterion(output2, target_model_label)  # 希望生成的假数据能让D判成1

                KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
                errG_KLD = torch.sigmoid(torch.sum(KLD_element))

                errG = errG1.add_(errG_KLD)
                errG.backward()

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
                # pass
                # # Save Losses for plotting later
                # G_losses.append(errG.item())
                # D_losses.append(errD.item())

                # # Save D(X) and D(G(z)) for plotting later
                # D_x_list.append(D_x)
                # D_z_list.append(D_G_z2)

                # Save the Best Model
                # if errG < loss_tep1 and epoch > 10:
                #     torch.save(netG.state_dict(), './results/VAE_Mnist2/model_errG.pt')
                #     loss_tep1 = errG
                if epoch % 10 == 0:
                    torch.save(self.model.generator.state_dict(), '/home/xlju/Project/ModelSimulator/output/gen_model' + str(epoch) + ".pt")



