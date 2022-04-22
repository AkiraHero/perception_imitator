from trainer.trainer_base import TrainerBase
import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms
from utils.loss import CustomLoss, SmoothL1Loss
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from utils.postprocess import non_max_suppression

class BaselineTrainer(TrainerBase):
    def __init__(self, config):
        super(BaselineTrainer, self).__init__()
        self.max_epoch = config['epoch']
        self.optimizer_config = config['optimizer']
        self.device = torch.device(config['device'])
        self.perception_loss_func = CustomLoss(config['loss_function'])
        self.prediction_loss_func = SmoothL1Loss()
        self.tensorboard_out_path = config['tensorboard_out_path']
        self.data_distributed = config['FP_distribution']
        self.optimizer = None
        self.data_loader = None
        
        pass

    def train_filter_pred(self, pred):
        if len(pred.size()) == 4:
            if pred.size(0) == 1:
                pred.squeeze_(0)
            else:
                raise ValueError("Tensor dimension is not right")

        cls_pred = pred[0, ...]
        activation = cls_pred > 0.1     # cls阈值
        num_boxes = int(activation.sum())

        if num_boxes == 0:
            #print("No bounding box found")
            return [], []

        corners = torch.zeros((num_boxes, 8))
        if self.data_distributed == True:
            for i in range(12, 20):
                corners[:, i - 12] = torch.masked_select(pred[i, ...], activation)
        else:
            for i in range(7, 15):
                corners[:, i - 7] = torch.masked_select(pred[i, ...], activation)
        corners = corners.view(-1, 4, 2).numpy()
        scores = torch.masked_select(cls_pred, activation).cpu().numpy()

        # NMS
        selected_ids = non_max_suppression(corners, scores, 0.2)        # iou阈值
        corners = corners[selected_ids]
        scores = scores[selected_ids]

        return corners, scores

    def get_batch_actor_features_and_match_list(self, pred, features):
        pred_match_list = []
        for batch_id in range(pred.shape[0]):
            per_pred = pred[batch_id].squeeze_(0)

            # Filter Predictions
            corners, scores = self.train_filter_pred(per_pred)

            # gt_boxes = np.array(label_list)       # 暂时还未写label_list的获取，后续需要用到pred_match用于轨迹匹配
            # gt_match, pred_match, overlaps = compute_matches(gt_boxes,
            #                                     corners, scores, iou_threshold=0.5)

            if len(corners) == 0:
                pass
            else:
                box_centers = np.mean(corners, axis=1)

                center_index = - box_centers / 0.2      # 0.2为resolution
                center_index[:, 0] += input.shape[-2]
                center_index[:, 1] += input.shape[-1] / 2
                center_index = np.round(center_index / 4).astype(int)        # 4为input_size/feature_size

                center_index = np.swapaxes(center_index, 1, 0)

                per_actor_features = features[batch_id, :, center_index[0], center_index[1]].permute(1, 0)

                if 'batch_actor_features' not in locals().keys():
                    batch_actor_features = per_actor_features
                else:
                    batch_actor_features = torch.cat((batch_actor_features, per_actor_features), 0)

                ######### !!!!还需要加入pred_match的列表用于匹配真值
        if 'batch_actor_features' not in locals().keys():   # 说明该batch中没有检测出任何物体
            return None, None
        else:
            return batch_actor_features, pred_match_list

    def set_optimizer(self, optimizer_config):
        optimizer_ref = torch.optim.__dict__[self.optimizer_config['type']]
        self.optimizer = optimizer_ref(self.model.parameters(), **optimizer_config['paras'])

    def run(self):
        if not self.check_ready():
            raise ModuleNotFoundError("The trainer not ready. Plz set model/dataset first")
        torch.autograd.set_detect_anomaly(True)
        self.set_optimizer(self.optimizer_config)
        self.model.set_device(self.device)
        self.perception_loss_func.to(self.device)
        self.prediction_loss_func.to(self.device)
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
                HDmap = data['HDmap']

                label_map = data['label_map'].permute(0, 3, 1, 2)
                
                ####################
                # Train perception #
                ####################
                input = torch.cat((occupancy, occlusion), dim=1)    # 将场景描述共同输入
                pred, features = self.model(input)
                perc_loss, cls, loc = self.perception_loss_func(pred, label_map)

                ####################
                # Train pridiction #
                ####################
                batch_actor_features, pred_match_list = self.get_batch_actor_features_and_match_list(pred, features)

                if batch_actor_features == None:
                    # loss = perc_loss
                    pass
                else:
                    way_points = self.model.prediction(batch_actor_features)    # 生成6个点的waypoints(6*2)
                    gt_way_points = ###########
                    pred_loss = self.prediction_loss_func(way_points, gt_way_points, pred_match_list)
                    print(pred_loss.item())

                loss = perc_loss + pred_loss
                loss.backward()
                self.optimizer.step()

                writer.add_scalar("loss_all", loss, self.global_step)
                writer.add_scalar("loss_perc", perc_loss, self.global_step)
                writer.add_scalar("cls", cls, self.global_step)
                writer.add_scalar("loc", loc, self.global_step)
                writer.add_scalar("loss_pred", pred_loss, self.global_step)

                print(
                        f'Epoch: [{epoch + 1:0>{len(str(epoch))}}/{self.max_epoch}]',
                        f'Step: [{step}/{len(self.data_loader)}]',
                        f'Loss-All: {loss:.4f}',
                        f'Loss-Perception: {perc_loss:.4f}',
                        f'cls: {cls:.4f}',
                        f'loc: {loc:.4f}',
                        f'Loss-Prediction: {pred_loss:.4f}',
                    )

            if epoch % 2 == 0:
                torch.save(self.model.state_dict(), \
                            'D:/1Pjlab/ADModel_Pro/output/baseline/' + str(epoch) + ".pt")



