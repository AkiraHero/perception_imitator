from trainer.trainer_base import TrainerBase
import torch.nn as nn
import torch
from torchvision import transforms
import logging

class FpExplicitTrainer(TrainerBase):
    def __init__(self, config):
        super(FpExplicitTrainer, self).__init__()
        self.max_epoch = config['epoch']
        self.optimizer_config = config['optimizer']
        self.device = torch.device(config['device'])
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

        # 初始化Optimizers和损失函数
        criterion = nn.BCELoss()  # Initialize BCELoss function

        # Training Loop
        # Lists to keep track of progress

        self.global_step = 0

        for epoch in range(self.max_epoch):
            self.epoch = epoch
            for step, data in enumerate(self.data_loader):
                self.step = step
                self.global_step += 1
                self.optimizer.zero_grad()

                explicit_data = data[:,:-1].float().to(device=self.device)
                label = torch.eye(2)[data[:,-1].long(),:].to(device=self.device)

                pred = self.model(explicit_data)

                loss = criterion(pred, label)
                loss.backward()
                self.optimizer.step()

                print(f'[loss] Epoch={epoch}/{self.max_epoch}, step={step}/{len(self.data_loader)}\t'
                                f'loss={loss:.6f}\t')

                # print current status and logging
                logging.info(f'[loss] Epoch={epoch}/{self.max_epoch}, step={step}/{len(self.data_loader)}\t'
                                f'loss={loss:.6f}\t'
                                )
                self.logger.log_data(loss.item(), True)

                if epoch % 10 == 0:
                    torch.save(self.model.state_dict(), 'D:/1Pjlab/ModelSimulator/output/fp_explicit_model/' + str(epoch) + ".pt")



