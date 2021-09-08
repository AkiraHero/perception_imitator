import logging

from trainer.trainer_base import TrainerBase
import torch.nn as nn
import torch
import numpy as np
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
        self.max_obj = 25
        pass

    def set_optimizer(self, optimizer_config):
        optimizer_ref = torch.optim.__dict__[self.optimizer_config[0]['type']]
        self.generator_optimizer = optimizer_ref(self.model.generator.parameters(), **optimizer_config[0]['paras'])
        optimizer_ref = torch.optim.__dict__[self.optimizer_config[1]['type']]
        self.discriminator_optimizer = optimizer_ref(self.model.discriminator.parameters(),
                                                     **optimizer_config[1]['paras'])

    def fullfill_obj(self, objs, max_obj_num):
        shp = objs.shape
        if len(objs.shape) != 3:
            raise TypeError("obj vector must have shape of 3: batchsize, num, [7dimboxes+class]")
        if shp[1] > max_obj_num:
            logging.warning(f"objsize({shp[1]}) larger than fixed max_obj_num({max_obj_num})!")
        new_shp = (shp[0], max_obj_num, shp[2])
        new_objs = torch.zeros(new_shp, device=objs.device)
        new_objs[:, :shp[1], :] = objs[:, :self.max_obj, :]
        return new_objs

    # todo move to online pvrcn

    # def lidar2camera(self, pvrcnn_res, calib):
    #     # pred_scores = box_dict['pred_scores'].cpu().numpy()
    #     # pred_boxes = box_dict['pred_boxes'].cpu().numpy()
    #     # pred_labels = box_dict['pred_labels'].cpu().numpy()
    #     # pred_dict = get_template_prediction(pred_scores.shape[0])
    #     # if pred_scores.shape[0] == 0:
    #     #     return pred_dict
    #     #
    #     # calib = batch_dict['calib'][batch_index]
    #     # image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
    #     # pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
    #     # pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
    #     #     pred_boxes_camera, calib, image_shape=image_shape
    #     # )
    #     #
    #     # pred_dict['name'] = np.array(class_names)[pred_labels - 1]
    #     # pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
    #     # pred_dict['bbox'] = pred_boxes_img
    #     # pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
    #     # pred_dict['location'] = pred_boxes_camera[:, 0:3]
    #     # pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
    #     # pred_dict['score'] = pred_scores
    #     # pred_dict['boxes_lidar'] = pred_boxes
    #
    #     """
    #     :param boxes3d_lidar: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    #     :param calib:
    #     :return:
    #         boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    #     """
    #     boxes3d_lidar = pvrcnn_res
    #     xyz_lidar = boxes3d_lidar[:, 0:3]
    #     l, w, h = boxes3d_lidar[:, 3:4], boxes3d_lidar[:, 4:5], boxes3d_lidar[:, 5:6]
    #     r = boxes3d_lidar[:, 6:7]
    #     c = boxes3d_lidar[:, 7:8]
    #     xyz_lidar[:, 2] -= h.reshape(-1) / 2
    #     pts_lidar_hom = torch.cat([xyz_lidar, torch.ones([xyz_lidar.shape[0], 1], device=xyz_lidar.device)], dim=1)
    #     trans = torch.from_numpy(np.dot(calib.V2C.T, calib.R0.T)).to(xyz_lidar.device)
    #     xyz_cam = pts_lidar_hom.mm(trans)
    #     r = -r - np.pi / 2
    #     return torch.cat([xyz_cam, l, h, w, r, c], axis=-1)

    # def process_pvrcnn_res(self, target_res, calib):
    #     cat_list = []
    #     for i, calib_ in zip(target_res, calib):
    #         cat_ = torch.cat([i['pred_boxes'], i['pred_labels'].reshape(-1, 1)], dim=1)
    #         cat_ = self.lidar2camera(cat_, calib_)
    #         new_cat_ = torch.zeros([self.max_obj, cat_.shape[1]], device=cat_.device)
    #         new_cat_[:cat_.shape[0], :] = cat_[:self.max_obj, :]
    #         new_cat_ = new_cat_.unsqueeze(0)
    #         cat_list.append(new_cat_)
    #
    #     # todo trans to camera coordinate
    #     return torch.cat(cat_list, 0)

    # # ?? matching shoube completed offline
    # def align_boxes(self, gt_boxes, target_boxes):
    #     if len(gt_boxes.shape) != 3 or len(target_boxes.shape) != 3:
    #         raise TypeError("obj vector must have shape of 3: batchsize, num, [7dimboxes+class]")
    #     if gt_boxes.shape != target_boxes.shape:
    #         raise TypeError("inputs must have same shape")
    #     return NotImplementedError

    def pre_process_gt_box(self, gt_boxes):
        if len(gt_boxes.shape) != 3:
            raise TypeError("obj vector must have shape of 3: batchsize, num, [7dimboxes+class]")
        gt = torch.zeros([gt_boxes.shape[0], self.max_obj, 8])
        gt[:, :gt_boxes.shape[1], :] = gt_boxes[:, :gt_boxes.shape[1], :]
        return gt

    def run(self):
        if not self.check_ready():
            raise ModuleNotFoundError("The trainer not ready. Plz set model/dataset first")
        torch.autograd.set_detect_anomaly(True)
        self.set_optimizer(self.optimizer_config)
        self.model.set_device(self.device)
        self.data_loader = self.dataset.get_data_loader()
        # Training Loop
        self.global_step = 0
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            for step, data in enumerate(self.data_loader):
                # zero grad
                self.model.discriminator.zero_grad()
                self.model.generator.zero_grad()

                # 0.data preparation
                # trans all data to gpu device
                self.data_loader.dataset.load_data_to_gpu(data)

                # get target model output
                target_res = self.model.target_model(data)
                target_boxes = target_res['dt_lidar_box']

                # align the gt_boxes and target_res_processed
                gt_box = self.pre_process_gt_box(data['gt_boxes'])
                gt_valid_mask = (gt_box[:, :, -1] > 0).to(self.device)
                gt_valid_elements = gt_valid_mask.sum()
                if not gt_valid_elements > 0:
                    raise ZeroDivisionError("wrong gt valid number")

                if gt_box.shape != target_boxes.shape:
                    raise TypeError("gt_box and target_box must have same shape")

                # 1.update discriminator

                # input data and gt_boxes as generator input / get generator output
                generator_input = data['points']
                generator_output, point_feature = self.model.generator(generator_input)

                # input generator output and data to discriminator
                discriminator_input_fake = {
                    "feature": point_feature.squeeze(-1),
                    "boxes": generator_output.detach()
                }
                out_d_fake = self.model.discriminator(discriminator_input_fake['feature'], discriminator_input_fake['boxes'])
                # input target_model output and data to discriminator
                discriminator_input_real = {
                    "feature": point_feature.squeeze(-1),
                    "boxes": target_boxes
                }
                out_d_real = self.model.discriminator(discriminator_input_real['feature'], discriminator_input_real['boxes'])

                # get discriminator loss
                # todo: need select valid object here
                gt_valid_mask = gt_valid_mask.unsqueeze(-1).repeat(1, 1, out_d_real.shape[2])
                assert gt_valid_mask.shape == out_d_fake.shape == out_d_real.shape
                err_fake = out_d_fake.mul(gt_valid_mask).sum() / gt_valid_elements
                err_real = - out_d_real.mul(gt_valid_mask).sum() / gt_valid_elements
                err_discriminator = err_fake + err_real

                # update discriminator
                err_discriminator.backward()
                self.discriminator_optimizer.step()

                # 2.update generator

                # encoding - sampling - generator again
                generator_output_2nd, point_feature_2nd = self.model.generator(generator_input)
                discriminator_input_fake_2nd = {
                    "feature": point_feature_2nd.squeeze(-1),
                    "boxes": generator_output_2nd
                }

                # discriminator judge and update generator
                out_d_fake_2nd = self.model.discriminator(discriminator_input_fake_2nd['feature'], discriminator_input_fake_2nd['boxes'])
                err_discriminator_2nd = -out_d_fake_2nd.mul(gt_valid_mask).sum() / gt_valid_elements
                err_discriminator_2nd.backward()
                self.generator_optimizer.step()

                # print current status and logging
                logging.info(f'[loss] Epoch={epoch}/{self.max_epoch}, step={step}/{len(self.data_loader)}\t'
                             f'D_fake={err_fake:.6f}\t'
                             f'D_real={err_real:.6f}\t'
                             f'D_total={err_discriminator:.6f}\t'
                             f'G_fake={err_discriminator_2nd:.6f}')
                self.logger.log_data("D_fake", err_fake.item(), True)
                self.logger.log_data("D_real", err_real.item(), True)
                self.logger.log_data("G_fake", err_discriminator_2nd.item(), True)


                self.step = step
                self.global_step += 1
            self.logger.log_model_params(self.model)
