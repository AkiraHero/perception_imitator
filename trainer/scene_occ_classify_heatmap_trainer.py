from trainer.trainer_base import TrainerBase
import torch.nn as nn
import torch
from torchvision import transforms
from utils.loss import CustomLoss
from tensorboardX import SummaryWriter

class SceneOccClassifyHeatmapTrainer(TrainerBase):
    def __init__(self, config):
        super(SceneOccClassifyHeatmapTrainer, self).__init__()
        self.max_epoch = config['epoch']
        self.optimizer_config = config['optimizer']
        self.device = torch.device(config['device'])
        self.loss_func = CustomLoss(config['loss_function'])
        self.tensorboard_out_path = config['tensorboard_out_path']
        self.optimizer = None
        self.data_loader = None
        
        pass

    def set_optimizer(self, optimizer_config):
        optimizer_ref = torch.optim.__dict__[self.optimizer_config['type']]
        self.optimizer = optimizer_ref(self.model.parameters(), **optimizer_config['paras'])

    def run(self):
        if not self.check_ready():
            raise ModuleNotFoundError("The trainer not ready. Plz set model/dataset first")
        torch.autograd.set_detect_anomaly(True)
        self.set_optimizer(self.optimizer_config)
        self.model.set_device(self.device)
        self.loss_func.to(self.device)
        self.data_loader = self.dataset.get_data_loader()
        writer = SummaryWriter(log_dir=self.tensorboard_out_path)

        # Training Loop
        self.global_step = 0
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            for step, data in enumerate(self.data_loader):
                self.step = step
                self.global_step += 1
                self.dataset.load_data_to_gpu(data)
            
                self.model.zero_grad()

                occupancy = data['occupancy'].unsqueeze(1)
                occlusion = data['occlusion'].unsqueeze(1)
                label_map = data['label_map'].permute(0, 3, 1, 2)
                
                input = torch.cat((occupancy, occlusion), dim=1)    # 将场景描述共同输入
                pred = self.model(input)
                loss, cls, loc = self.loss_func(pred, label_map)

                loss.backward()
                self.optimizer.step()

                writer.add_scalar("loss_all", loss, self.global_step)
                writer.add_scalar("cls", loss, self.global_step)
                writer.add_scalar("loc", loss, self.global_step)

                print(
                        f'Epoch: [{epoch + 1:0>{len(str(epoch))}}/{self.max_epoch}]',
                        f'Step: [{step}/{len(self.data_loader)}]',
                        f'Loss-All: {loss:.4f}',
                        f'cls: {cls:.4f}',
                        f'loc: {loc:.4f}',
                    )
 
            if epoch % 2 == 0:
                torch.save(self.model.state_dict(), \
                            'D:/1Pjlab/ADModel_Pro/output/scene_occ_heatmap/' + str(epoch) + ".pt")



