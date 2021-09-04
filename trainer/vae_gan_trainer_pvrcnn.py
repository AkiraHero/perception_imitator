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
        self.max_obj = 10
        pass

    def set_optimizer(self, optimizer_config):
        optimizer_ref = torch.optim.__dict__[self.optimizer_config[0]['type']]
        self.generator_optimizer = optimizer_ref(self.model.generator.parameters(), **optimizer_config[0]['paras'])
        optimizer_ref = torch.optim.__dict__[self.optimizer_config[1]['type']]
        self.discriminator_optimizer = optimizer_ref(self.model.discriminator.parameters(), **optimizer_config[1]['paras'])


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

    def lidar2camera(self, pvrcnn_res, calib):
        # pred_scores = box_dict['pred_scores'].cpu().numpy()
        # pred_boxes = box_dict['pred_boxes'].cpu().numpy()
        # pred_labels = box_dict['pred_labels'].cpu().numpy()
        # pred_dict = get_template_prediction(pred_scores.shape[0])
        # if pred_scores.shape[0] == 0:
        #     return pred_dict
        #
        # calib = batch_dict['calib'][batch_index]
        # image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
        # pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
        # pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
        #     pred_boxes_camera, calib, image_shape=image_shape
        # )
        #
        # pred_dict['name'] = np.array(class_names)[pred_labels - 1]
        # pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
        # pred_dict['bbox'] = pred_boxes_img
        # pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
        # pred_dict['location'] = pred_boxes_camera[:, 0:3]
        # pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
        # pred_dict['score'] = pred_scores
        # pred_dict['boxes_lidar'] = pred_boxes

        """
        :param boxes3d_lidar: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
        :param calib:
        :return:
            boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
        """
        boxes3d_lidar = pvrcnn_res
        xyz_lidar = boxes3d_lidar[:, 0:3]
        l, w, h = boxes3d_lidar[:, 3:4], boxes3d_lidar[:, 4:5], boxes3d_lidar[:, 5:6]
        r = boxes3d_lidar[:, 6:7]
        c = boxes3d_lidar[:, 7:8]
        xyz_lidar[:, 2] -= h.reshape(-1) / 2
        pts_lidar_hom = torch.cat([xyz_lidar, torch.ones([xyz_lidar.shape[0], 1], device=xyz_lidar.device)], dim=1)
        trans = torch.from_numpy(np.dot(calib.V2C.T, calib.R0.T)).to(xyz_lidar.device)
        xyz_cam = pts_lidar_hom.mm(trans)
        r = -r - np.pi / 2
        return torch.cat([xyz_cam, l, h, w, r, c], axis=-1)


    def process_pvrcnn_res(self, target_res, calib):
        cat_list = []
        for i, calib_ in zip(target_res, calib):
            cat_ = torch.cat([i['pred_boxes'], i['pred_labels'].reshape(-1, 1)], dim=1)
            cat_ = self.lidar2camera(cat_, calib_)
            new_cat_ = torch.zeros([self.max_obj, cat_.shape[1]], device=cat_.device)
            new_cat_[:cat_.shape[0], :] = cat_[:self.max_obj, :]
            new_cat_ = new_cat_.unsqueeze(0)
            cat_list.append(new_cat_)

        # todo trans to camera coordinate

        return torch.cat(cat_list, 0)

    # ?? matching shoube completed offline
    def align_boxes(self, gt_boxes, target_boxes):
        if len(gt_boxes.shape) != 3 or len(target_boxes.shape) != 3:
            raise TypeError("obj vector must have shape of 3: batchsize, num, [7dimboxes+class]")
        if gt_boxes.shape != target_boxes.shape:
            raise TypeError("inputs must have same shape")
        return NotImplementedError



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

                # trans all data to gpu device
                self.data_loader.dataset.load_data_to_gpu(data)

                # fullfill the objlist to a vector with unified size
                gt_boxes = self.fullfill_obj(data['gt_boxes'], self.max_obj)

                # get target model output
                target_res = self.model.target_model(data)
                # concatenate batch_box_preds and batch_cls_preds
                # resize/fullfill the objlist(target model) to a vector with unified size
                target_boxes = self.process_pvrcnn_res(target_res[0], data['calib'])

                # align the gt_boxes and target_res_processed



                target_boxes = self.fullfill_obj(data['pred_boxes'])

                # get data encoding

                # concatenate data encoding and gt_boxes as generator input

                # get generator output

                # concatenate generator output/ target model output with data features

                # discriminator judge

                # update discriminator

                # encoding - sampling - generator again

                # discriminator judge and update generator






                self.step = step
                self.global_step += 1
                pass



