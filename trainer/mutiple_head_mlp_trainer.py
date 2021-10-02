from trainer.trainer_base import TrainerBase
import torch
import logging


class MultipleHeadMLPTrainer(TrainerBase):
    def __init__(self, config):
        super(MultipleHeadMLPTrainer, self).__init__()
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

        # Training Loop
        # Lists to keep track of progress

        self.global_step = 0

        for epoch in range(self.max_epoch):
            self.epoch = epoch
            for step, data in enumerate(self.data_loader):
                self.optimizer.zero_grad()
                self.dataset.load_data2gpu(data)
                self.model(data)
                # tips:
                # if loss can be decided by model self without considering the training logistics
                # please calculate/set loss in model
                loss = self.model.get_loss()
                loss['loss'].backward()

                # print current status and logging
                if not self.distributed or self.rank == 0:
                    logging.info(f'[loss] Epoch={epoch}/{self.max_epoch}, step={step}/{len(self.data_loader)}\t'
                                 f'loss={loss["loss"]:.6f}\t'
                                 )
                    for i in loss.keys():
                        self.logger.log_data(i, loss[i].item(), True)
                self.step = step
                self.global_step += 1
            if not self.distributed or self.rank == 0:
                self.logger.log_model_params(self.model)




