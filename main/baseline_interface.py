import os
import math
import cv2
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
from factory.model_factory import ModelFactory
from factory.dataset_factory import DatasetFactory
from utils.config.Configuration import Configuration
from utils.postprocess import *
from utils.preprocess import two_points_2_line, two_points_distance, euclidean_distance

import warnings
warnings.filterwarnings("ignore")

class BaselineInterface:
    def __init__(self, ckpt_file, cfg_dir):
        self._config = Configuration()
        self._cfg_dir = cfg_dir
        self.update_config()
        self._device = self._config.training_config["device"]  

        self._model = ModelFactory.get_model(self._config.model_config)
        self.set_model(ckpt_file)

        self._dataset = DatasetFactory.get_dataset(self._config.dataset_config)
        self._data_loader = self._dataset.get_data_loader()

    def update_config(self):
        args = self._config.get_shell_args_train()
        args.cfg_dir = self._cfg_dir
        args.for_train = False
        args.shuffle = False
        args.interface = True
        self._config.load_config(args.cfg_dir)
        self._config.overwrite_config_by_shell_args(args)

    def set_model(self, checkpoint_file):
        paras = torch.load(checkpoint_file)
        self._model.load_model_paras(paras)
        self._model.set_decode(True)
        self._model.set_eval()
        self._model.set_device(self._device)

    def get_model_input(self, input_list, HD_map):
        pic_height, pic_width = self._dataset._geometry['label_shape'][:2]
        ratio = self._dataset._geometry['ratio']
        img_cam_pos = [pic_height, pic_width/2]
        occupancy = np.zeros([pic_height, pic_width]).astype('int32')
        occlusion = np.ones([pic_height, pic_width]).astype('int32')

        for one_bbox in input_list:
            ############################
            # First stage: get occupancy
            ############################
            lidar_x = one_bbox[0]
            lidar_y = one_bbox[1]
            lidar_l = one_bbox[2]
            lidar_w = one_bbox[3]
            theta = one_bbox[4]

            # 调整坐标到图片坐标系
            img_x, img_y = self._dataset.transform_metric2label(np.array([[lidar_x, lidar_y]]))[0].astype('int32')
            img_l = lidar_l / ratio
            img_w = lidar_w / ratio

            # get occupancy
            # 初步筛选，减小计算算量
            pass_size = int(np.ceil(0.5*math.sqrt(img_l**2 + img_w**2)))

            pix_x_min = max(0, img_x - pass_size)
            pix_x_max = min(pic_height, img_x + pass_size)
            pix_y_min = max(0, img_y - pass_size)
            pix_y_max = min(pic_width, img_y + pass_size)

            for pix_x in range(pix_x_min, pix_x_max):
                for pix_y in range(pix_y_min, pix_y_max):
                    w_dis = euclidean_distance(np.tan(theta + math.pi), img_y - np.tan(theta + math.pi)*img_x, [pix_x, pix_y])
                    l_dis = euclidean_distance(-1/np.tan(theta + math.pi), img_y + 1/np.tan(theta + math.pi)*img_x, [pix_x, pix_y])  

                    if w_dis <= img_w / 2 and l_dis <= img_l / 2:
                        occupancy[pix_x, pix_y] = 1

            ############################                
            # Second stage: get occlusion
            ############################
            corners, _ = self._dataset.get_corners([lidar_x, lidar_y, lidar_l, lidar_w, theta], use_distribution=False)
            label_corners = self._dataset.transform_metric2label(corners)

            k_list = []
            b_list = [] 
            dist_list = []
            intersection_points = []
            
            if (label_corners < 0).sum() > 0 or (label_corners[:, 0] > 352).sum() == 4 or (label_corners[:, 1] > 352).sum() == 4:
                continue
            
            for i in range(label_corners.shape[0]):
                # 需要截断超出bev视图的部分
                label_corners[i][0] = max(0, label_corners[i][0])  
                label_corners[i][0] = min(352, label_corners[i][0])
                label_corners[i][1] = max(0, label_corners[i][1])  
                label_corners[i][1] = min(400, label_corners[i][1])

                k, b = two_points_2_line(label_corners[i], img_cam_pos)
                dist = two_points_distance(label_corners[i], img_cam_pos)

                if k != float("-inf") and  k != float("inf"): 
                    if b < 0:
                        interp_y = 0
                        interp_x = - b / k
                    elif b >= 400:
                        interp_y = 400
                        interp_x = (interp_y - b) / k
                    else:
                        interp_x = 0
                        interp_y = b
                else:
                    if label_corners[i][1] <= 200:
                        b = -1
                        interp_x = 352
                        interp_y = 0
                    else:
                        b =400 + 1
                        interp_x = 352
                        interp_y = 400
                interp = (interp_x, interp_y)

                k_list.append(k)
                b_list.append(b)
                dist_list.append(dist)
                intersection_points.append(interp)

            min_dist_index = dist_list.index(min(dist_list))

            for i in range(0, 3):
                points = []
                if i >= min_dist_index:
                    if (b_list[i+1] < 0 and 0 <= b_list[min_dist_index] < 400) or (b_list[min_dist_index] < 0 and 0 <= b_list[i+1] < 400):
                        points = [label_corners[i+1], label_corners[min_dist_index], intersection_points[min_dist_index], [0,0], intersection_points[i+1]]
                    elif (b_list[i+1] >= 400 and 0 <= b_list[min_dist_index] < 400) or (b_list[min_dist_index] >= 400 and 0 <= b_list[i+1] < 400):
                        points = [label_corners[i+1], label_corners[min_dist_index], intersection_points[min_dist_index], [0,400], intersection_points[i+1]]
                    elif b_list[i+1] >= 400 and b_list[min_dist_index] < 0:
                        points = [label_corners[i+1], label_corners[min_dist_index], intersection_points[min_dist_index], [0,0], [0,400], intersection_points[i+1]]
                    elif b_list[min_dist_index] >= 400 and b_list[i+1] < 0:
                        points = [label_corners[i+1], label_corners[min_dist_index], intersection_points[min_dist_index], [0,400], [0,0], intersection_points[i+1]]
                    else:
                        points = [label_corners[i+1], label_corners[min_dist_index], intersection_points[min_dist_index], intersection_points[i+1]]
                else:
                    if (b_list[i] < 0 and 0 <= b_list[min_dist_index] < 400) or (b_list[min_dist_index] < 0 and 0 <= b_list[i] < 400):
                        points = [label_corners[i], label_corners[min_dist_index], intersection_points[min_dist_index], (0,0), intersection_points[i]]
                    elif (b_list[i] >= 400 and 0 <= b_list[min_dist_index] < 400) or (b_list[min_dist_index] >= 400 and 0 <= b_list[i] < 400):
                        points = [label_corners[i], label_corners[min_dist_index], intersection_points[min_dist_index], [0,400], intersection_points[i]]
                    elif b_list[i] >= 400 and b_list[min_dist_index] < 0:
                        points = [label_corners[i], label_corners[min_dist_index], intersection_points[min_dist_index], [0,0], [0,400], intersection_points[i]]
                    elif b_list[min_dist_index] >= 400 and b_list[i] < 0:
                        points = [label_corners[i], label_corners[min_dist_index], intersection_points[min_dist_index], [0,400], [0,0], intersection_points[i]]
                    else:
                        points = [label_corners[i], label_corners[min_dist_index], intersection_points[min_dist_index], intersection_points[i]]

                points = np.array(points).reshape(-1,1,2).astype(np.int32)
                matrix = np.zeros((pic_width, pic_height), dtype=np.int32)
                cv2.drawContours(matrix, [points], -1, (1), thickness=-1)
                list_of_points_indices = np.nonzero(np.swapaxes(matrix, 1, 0))
                occlusion[list_of_points_indices] = 0

        occupancy = torch.from_numpy(occupancy).unsqueeze(0)
        occlusion = torch.from_numpy(occlusion).unsqueeze(0)
        HDmap = torch.from_numpy(HD_map).permute(2, 0, 1)

        # get input
        input = torch.cat((occupancy, occlusion, HDmap), dim=0).float().to(self._device)

        return input

    def __call__(self, input_list, HD_map):
        with torch.no_grad():
            self.model_input = self.get_model_input(input_list, HD_map)

            # Forward Detection
            pred, features = self._model(self.model_input.unsqueeze(0))
            pred.squeeze_(0)

            corners, scores = filter_pred(self._config, pred)

            return corners