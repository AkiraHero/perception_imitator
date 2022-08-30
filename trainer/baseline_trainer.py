from tkinter import N
from trainer.trainer_base import TrainerBase
import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms
from utils.loss import CustomLoss, SmoothL1Loss
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from utils.postprocess import non_max_suppression, compute_matches

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
        self.actor_feature_size = config['actor_feature_size']
        self.optimizer = None
        self.data_loader = None
        
        pass

    def train_filter_pred(self, pred, decoded_pred):
        if len(pred.size()) == 4:
            if pred.size(0) == 1:
                pred = pred.squeeze(0)
            else:
                raise ValueError("Tensor dimension is not right")

        cls_pred = pred[0, ...]
        activation = cls_pred > 0.2     # cls阈值
        num_boxes = int(activation.sum())

        if num_boxes == 0:
            #print("No bounding box found")
            return [], []

        corners = torch.zeros((num_boxes, 8))
        for i in range(0, 8):
            corners[:, i] = torch.masked_select(decoded_pred[i, ...], activation)
        corners = corners.view(-1, 4, 2).numpy()
        scores = torch.masked_select(cls_pred, activation).cpu().numpy()

        # NMS
        selected_ids = non_max_suppression(corners, scores, 0.3)        # iou阈值
        corners = corners[selected_ids]
        scores = scores[selected_ids]

        return corners, scores

    def get_batch_actor_features_and_match_list(self, pred, decoded_pred, features, label_list):
        pred_match_list = []
        aug_pred_match_list = []
        for batch_id in range(pred.shape[0]):
            per_pred = pred[batch_id].squeeze(0)
            per_decoded_pred = decoded_pred[batch_id].squeeze(0)
            per_label_list = label_list[batch_id]

            # Filter Predictions
            corners, scores = self.train_filter_pred(per_pred, per_decoded_pred)

            gt_boxes = np.array(per_label_list)
            gt_match, pred_match, overlaps = compute_matches(gt_boxes,
                                                corners, scores, iou_threshold=0.5)                               
            pred_match_list.append(list(pred_match))

        ##################
        ### No Augment ###
        ##################
        #     if len(corners) == 0:
        #         pass
        #     else:
        #         box_centers = np.mean(corners, axis=1)

        #         center_index = - box_centers / 0.2      # 0.2为resolution
        #         center_index[:, 0] += pred.shape[-2]
        #         center_index[:, 1] += pred.shape[-1] / 2
        #         center_index = np.round(center_index / 4).astype(int)        # 4为input_size/feature_size，将坐标从原图转为特征图

        #         center_index = np.swapaxes(center_index, 1, 0)
        #         center_index[0] = np.clip(center_index[0], 0, features.shape[-2] - 1)
        #         center_index[1] = np.clip(center_index[1], 0, features.shape[-1] - 1)

        #         per_actor_features = features[batch_id, :, center_index[0], center_index[1]].permute(1, 0)

        #         if 'batch_actor_features' not in locals().keys():
        #             batch_actor_features = per_actor_features
        #         else:
        #             batch_actor_features = torch.cat((batch_actor_features, per_actor_features), 0)

        # if 'batch_actor_features' not in locals().keys():   # 说明该batch中没有检测出任何物体
        #     return None, None
        # else:
        #     return batch_actor_features, pred_match_list

        ###############
        ### Augment ###
        ###############
            aug_pred_match = []
            aug_per_actor_features = []
            
            for idx, corner in enumerate(corners):
                label_corner = self.dataset.transform_metric2label(corner)
                feature_corner = np.round(label_corner / 4).astype(int)
                points = self.dataset.get_points_in_a_rotated_box(feature_corner, list(features.shape[-2:]))
                aug_pred_match.extend(len(points) * [pred_match[idx]])

                for p in points:        # 将特征图中bbox围住的每一个点都作为正样本
                    label_x = min(p[0], features.shape[-2] - 1)
                    label_y = min(p[1], features.shape[-1] - 1)
                    per_actor_feature = features[batch_id, :, label_x, label_y][None,None,:]
                    aug_per_actor_feature = nn.functional.interpolate(per_actor_feature, size=self.actor_feature_size, mode='linear').squeeze(0)

                    aug_per_actor_features.append(aug_per_actor_feature)
            
            if len(aug_per_actor_features) == 0:
                pass
            else:
                aug_per_actor_features = torch.cat(aug_per_actor_features, dim=0)
                if 'aug_batch_actor_features' not in locals().keys():
                    aug_batch_actor_features = aug_per_actor_features
                else:
                    aug_batch_actor_features = torch.cat((aug_batch_actor_features, aug_per_actor_features), 0)

            aug_pred_match_list.append(list(aug_pred_match))

        if 'aug_batch_actor_features' not in locals().keys():   # 说明该batch中没有检测出任何物体
            return None, None
        else:
            return aug_batch_actor_features, aug_pred_match_list



    def set_optimizer(self, optimizer_config):
        optimizer_ref = torch.optim.__dict__[self.optimizer_config['type']]
        self.optimizer = optimizer_ref(self.model.parameters(), **optimizer_config['paras'])

    def run(self):
        if not self.check_ready():
            raise ModuleNotFoundError("The trainer not ready. Plz set model/dataset first")
        torch.autograd.set_detect_anomaly(True)
        self.set_optimizer(self.optimizer_config)
        self.model.set_device(self.device)

        
        # # For test
        # pretext_model = torch.load("C:/Users/Sunyyyy/Desktop/Study/PJLAB/Code/ADModel_Pro/output/baseline_kitti_range/50.pt")
        # model2_dict = self.model.state_dict()
        # state_dict = {k:v for k,v in pretext_model.items() if k in model2_dict.keys()}
        # model2_dict.update(state_dict)
        # self.model.load_model_paras(model2_dict)

        
        self.perception_loss_func.to(self.device)
        self.prediction_loss_func.to(self.device)
        self.data_loader = self.dataset.get_data_loader()
        # writer = SummaryWriter(log_dir=self.tensorboard_out_path)

        # Training Loop
        print("device: ", self.device)
        print("Start training!")
        self.global_step = 0
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            for step, data in enumerate(self.data_loader):
                self.step = step
                self.global_step += 1
                self.dataset.load_data_to_gpu(data)
            
                self.model.zero_grad()

                occupancy = data['occupancy'].permute(0, 3, 1, 2)
                occlusion = data['occlusion'].permute(0, 3, 1, 2)
                HDmap = data['HDmap'].permute(0, 3, 1, 2)
                label_map = data['label_map'].permute(0, 3, 1, 2)
                label_list = data['label_list']
                # waypoints = data['future_waypoints_st']

                ####################
                # Train perception #
                ####################
                input = torch.cat((occupancy, occlusion, HDmap), dim=1)    # 将场景描述共同输入
                # input = torch.cat((occupancy, occlusion), dim=1)    # 将场景描述共同输入
                pred, features = self.model(input)
                perc_loss, cls, loc, cls_loss = self.perception_loss_func(pred, label_map)

                if epoch < 30:   # 为了助于收敛，前三轮只对cls分支进行参数回传
                    loss = cls_loss
                else:
                    loss = perc_loss

                ####################
                # Train pridiction #
                ####################
                # decoded_pred = self.model.corner_decoder(pred.detach()[:, 1:,...])     # 将pred_map解码为可获取corner的形式
                # batch_actor_features, pred_match_list = self.get_batch_actor_features_and_match_list(pred.detach(), decoded_pred, features.detach(), label_list)

                # if epoch < 2:   # 为了助于收敛，前三轮只对cls分支进行参数回传
                #     loss = cls_loss
                #     pred_loss = np.NaN
                # else: 
                #     if pred_match_list == None or np.array(np.concatenate(pred_match_list, axis=0) >= 0).sum()==0:    # 检测结果未匹配上GT
                #         loss = perc_loss
                #         pred_loss = np.NaN
                #     else:
                #         pred_mask = np.array(np.concatenate(pred_match_list, axis=0) >= 0)      # 匹配上GT的mask

                #         filter_batch_actor_features = batch_actor_features[pred_mask]           # 只对匹配上GT的检测框中心点处的特征进行预测
                #         pred_way_points = self.model.prediction(filter_batch_actor_features)    # 预测6个点的waypoints(6*2)

                #         gt_way_points = []   # 根据匹配结果获取对应的真值
                #         for batch_id, one_match_list in enumerate(pred_match_list):
                #             index = [i for i in one_match_list if i >= 0]
                #             if len(index) == 0:
                #                 continue
                #             one_gt_way_points = waypoints[batch_id][index]

                #             gt_way_points.append(one_gt_way_points)
                        
                #         gt_way_points = torch.tensor(np.concatenate(gt_way_points, axis=0), dtype=torch.float32).view(-1, 12).cuda()

                #         pred_loss = self.prediction_loss_func(pred_way_points, gt_way_points)
                #         loss = perc_loss + pred_loss

                loss.backward()
                self.optimizer.step()

                # writer.add_scalar("loss_all", loss, self.global_step)
                # writer.add_scalar("loss_perc", perc_loss, self.global_step)
                # writer.add_scalar("cls", cls, self.global_step)
                # writer.add_scalar("loc", loc, self.global_step)
                # writer.add_scalar("loss_pred", pred_loss, self.global_step)

                print(
                    f'Epoch: [{epoch + 1:0>{len(str(epoch))}}/{self.max_epoch}]',
                    f'Step: [{step}/{len(self.data_loader)}]',
                    f'Loss-All: {loss.item():.4f}',
                    f'Loss-cls: {cls:.4f}',
                    f'Loss-loc: {loc:.4f}',
                    # f'Loss-Prediction: {pred_loss:.4f}',
                )

            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), \
                            './output/kitti/baseline_pp_pose/' + str(epoch) + ".pt")



