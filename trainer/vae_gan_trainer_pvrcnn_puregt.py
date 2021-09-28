import logging

from trainer.trainer_base import TrainerBase
import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms


class VAEGANTrainerPVRCNNPureGT(TrainerBase):
    def __init__(self, config):
        super(VAEGANTrainerPVRCNNPureGT, self).__init__()
        self.max_epoch = config['epoch']
        self.optimizer_config = config['optimizer']
        if not self.distributed:
            self.device = torch.device(config['device'])
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.data_loader = None
        self.max_obj = 25
        self.sync_bn = config['sync_bn']
        self.use_kld_loss = config['use_kld_loss']
        self.n_critic = config['n_critic']
        self.weight_clip = config['weight_clip']

    def set_optimizer(self, optimizer_config):
        model = self.model
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        optimizer_ref = torch.optim.__dict__[self.optimizer_config[0]['type']]
        self.generator_optimizer = optimizer_ref(model.generator.parameters(), **optimizer_config[0]['paras'])
        optimizer_ref = torch.optim.__dict__[self.optimizer_config[1]['type']]
        self.discriminator_optimizer = optimizer_ref(model.discriminator.parameters(),
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
        gt[:, :gt_boxes.shape[1], :] = gt_boxes[:, :self.max_obj, :]
        return gt

    def expand_box_as_d_input(self, boxes):
        box_vec = []
        for i in boxes:
            boxlist = [box for box in i]
            box_vec.append(torch.cat(boxlist, dim=0))
        return torch.stack(box_vec)

    def run(self):
        torch.autograd.set_detect_anomaly(True)
        super(VAEGANTrainerPVRCNNPureGT, self).run()
        model = self.model
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        loss_func = torch.nn.BCELoss()
        # Training Loop
        self.global_step = 0
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            for step, data in enumerate(self.data_loader):
                # zero grad
                model.discriminator.zero_grad()
                model.generator.zero_grad()

                # 0.data preparation
                # trans all data to gpu device
                self.data_loader.dataset.load_data_to_gpu(data)

                # get target model output
                target_res = model.target_model(data)
                target_boxes = target_res['dt_lidar_box']

                # align the gt_boxes and target_res_processed
                gt_box = self.pre_process_gt_box(data['gt_boxes'])
                # gt_valid_mask = (gt_box[:, :, -1] > 0).to(self.device)
                # gt_valid_elements = gt_valid_mask.sum()
                gt_box = gt_box.to(self.device)
                # if not gt_valid_elements > 0:
                #     raise ZeroDivisionError("wrong gt valid number")

                if gt_box.shape != target_boxes.shape:
                    raise TypeError("gt_box and target_box must have same shape")

                # 1.update discriminator

                # input data and gt_boxes as generator input / get generator output

                gt_box = self.pre_process_gt_box(data['gt_boxes']).to(self.device)
                generator_input = {
                    'boxes': gt_box
                }
                generator_output = model.generator(generator_input)

                # input generator output and data to discriminator

                out_d_fake = model.discriminator(torch.cat([self.expand_box_as_d_input(generator_output + gt_box), self.expand_box_as_d_input(gt_box)], dim=1))
                # input target_model output and data to discriminator
                out_d_real = model.discriminator(torch.cat([self.expand_box_as_d_input(target_boxes), self.expand_box_as_d_input(gt_box)], dim=1))

                # get discriminator loss
                # todo: need select valid object here

                err_fake = loss_func(out_d_fake, torch.zeros(out_d_fake.shape, device=self.device)) / out_d_fake.shape[0]
                err_real = loss_func(out_d_real, torch.ones(out_d_real.shape, device=self.device)) / out_d_real.shape[0]
                err_discriminator = err_fake + err_real

                # update discriminator
                err_discriminator.backward()
                self.discriminator_optimizer.step()
                for p in model.discriminator.parameters():
                    p.data.clamp_(-self.weight_clip, self.weight_clip)

                # 2.update generator

                # encoding - sampling - generator again

                if step % self.n_critic == 0:
                    generator_output_2nd = model.generator(generator_input)

                    # discriminator judge and update generator
                    out_d_fake_2nd = model.discriminator(torch.cat([self.expand_box_as_d_input(gt_box + generator_output_2nd), self.expand_box_as_d_input(gt_box)], dim=1))
                    err_discriminator_2nd = loss_func(out_d_fake_2nd,
                                                      torch.ones(out_d_fake_2nd.shape, device=self.device)) / out_d_fake_2nd.shape[0]
                    err_generator = 0.
                    err_generator += err_discriminator_2nd
                    err_generator.backward()
                    self.generator_optimizer.step()

                # print current status and logging: todo: distributed
                if self.rank == 0 or self.rank is None:
                    logging.info(f'[loss] Epoch={epoch}/{self.max_epoch}, step={step}/{len(self.data_loader)}\t'
                                 f'D_fake={err_fake:.6f}\t'
                                 f'D_real={err_real:.6f}\t'
                                 f'D_total={err_discriminator:.6f}\t'
                                 f'G_fake_d={err_discriminator_2nd:.6f}\t'
                                 f'G_fake={err_generator:.6f}'
                                 )
                    self.logger.log_data("D_fake", err_fake.item(), True)
                    self.logger.log_data("D_real", err_real.item(), True)
                    if step % self.n_critic == 0:
                        self.logger.log_data("G_fake_d", err_discriminator_2nd.item(), True)
                        self.logger.log_data("G_fake", err_generator.item(), True)

                self.step = step
                self.global_step += 1
            if self.rank == 0:
                self.logger.log_model_params(model)
