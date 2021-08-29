from trainer.trainer_base import TrainerBase
import torch.nn as nn
import torch
from torchvision import transforms


class VAEWGANTrainer(TrainerBase):
    def __init__(self, config):
        super(VAEGANTrainer, self).__init__()
        self.max_epoch = config['epoch']
        self.optimizer_config = config['optimizer']
        self.device = torch.device(config['device'])
        self.n_critic = config['n_critic']
        self.weight_clip = config['weight_clip']
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

        # data transform
        tf_normalize = transforms.Normalize(0.5, 0.5)
        tf_resize = transforms.Resize(14)
        for epoch in range(self.max_epoch):
            for step, data in enumerate(self.data_loader):
                imgs = data[0].to(device=self.device)
                # gt_labels = data[1]
                cur_batch_size = data[0].shape[0]
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

                G_score, mu, logvar = self.model.generator(generator_input)  # 得到每一类的得分
                G_score = G_score.unsqueeze(-2)

                discriminator_input_fake = torch.cat((flattened_img_vector, G_score), 2)
                errD = -torch.mean(self.model.discriminator(discriminator_input_real)) + \
                       torch.mean(self.model.discriminator(discriminator_input_fake.detach()))

                # # Update D
                errD.backward()
                self.discriminator_optimizer.step()
                for p in self.model.discriminator.parameters():
                    p.data.clamp_(-self.weight_clip, self.weight_clip)

                if step % self.n_critic == 0:
                    ############################
                    # (2) Update G network: maximize log(D(G(z)))
                    ###########################
                    self.model.generator.zero_grad()
                    new_G_score, mu, logvar = self.model.generator(generator_input)  # 得到每一类的得分
                    new_G_score = new_G_score.unsqueeze(-2)
                    new_discriminator_input_fake = torch.cat((flattened_img_vector, new_G_score), 2)
                    errG = -torch.mean(self.model.discriminator(new_discriminator_input_fake).view(-1))
                    errG.backward()
                    # Update G
                    self.generator_optimizer.step()

                print(
                    f'Epoch: [{epoch + 1:0>{len(str(epoch))}}/{self.max_epoch}]',
                    f'Step: [{step}/{len(self.data_loader)}]',
                    f'Loss-D: {errD.item():.4f}',
                    f'Loss-G: {errG.item():.4f}',
                )

                if epoch % 10 == 0:
                    torch.save(self.model.generator.state_dict(), '/home/xlju/Project/ModelSimulator/output/gen_model' + str(epoch) + ".pt")



