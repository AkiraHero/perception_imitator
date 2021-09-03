from trainer.trainer_base import TrainerBase
import torch.nn as nn
import torch
from torchvision import transforms

class VAEGANTrainerPVRCNN(TrainerBase):
    def __init__(self, config):
        super(VAEGANTrainerPVRCNN, self).__init__()
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


        # Training Loop
        # Lists to keep track of progress

        self.global_step = 0
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            for step, data in enumerate(self.data_loader):
                self.data_loader.dataset.load_data_to_gpu(data)
                target_res = self.model.target_model(data)


                self.step = step
                self.global_step += 1

                pass



