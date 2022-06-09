import os
import math
import cv2
import random
from cProfile import label
from cmath import pi
from optparse import Values
import pickle
from matplotlib.style import use
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.dataset_base import DatasetBase
from collections import defaultdict
from pyquaternion import Quaternion
from utils.preprocess import two_points_2_line, two_points_distance, euclidean_distance

'''
Dataset for nuScenes easy scene expression loading
    occupancy: H * W
    occlusion: H * W
    rasterized map: H * W * 3
    label_map: H * W * 7 , 7 channels are [cls_FP_or_not, cos(yaw), sin(yaw), x, y, log(w), log(l)]
    trajectory
    
'''

class ActorNoiseCarlaDataset(DatasetBase):
    
    def __init__(self, config):
        super(DatasetBase, self).__init__()
        self._is_train = config['paras']['for_train']
        self._batch_size = config['paras']['batch_size']
        self._data_root = config['paras']['data_root']
        self._num_workers = config['paras']['num_workers']
        self._shuffle = config['paras']['shuffle']
        self._town = config['paras']['Town']
        self._geometry = config['paras']['geometry']
        self._sweeps_len = config['paras']['sweeps_len']
        assert self._sweeps_len > 0 and isinstance(self._sweeps_len, int)
        self._target_model = config['paras']['target_model']
        self._is_interface = config['paras']['interface']
        self._distribution_setting = config['paras']['FP_distribution']
        self._car_std = config['paras']['waypoints_std']['car']
        self._pedestrian_std = config['paras']['waypoints_std']['pedestrian']
        
        self.label_path = os.path.join(self._data_root, "carla_new/label_2/")
        self.pose_path = os.path.join(self._data_root, "carla_new/pose/")

        gt_file = os.path.join("./data", "carla_%s_sim_model_gt.pkl" %self._target_model)
        with open(gt_file, 'rb') as f:
            self._gt_all = pickle.load(f)   # 加载target detection model和target prediction model下的真值
        random.seed(2021)
        random.shuffle(self._gt_all)
        offset = int(len(self._gt_all) * 0.8)
        if self._is_train == True:
            self._gt = self._gt_all[:offset]
        else:
            self._gt = self._gt_all[offset:]

        # 获取检测结果
        detect_result_file = os.path.join("./data", "carla_%s_match_gt.pkl" %self._target_model)
        with open(detect_result_file, 'rb') as f:     # 加载检测结果
            self.dt_results = pickle.load(f)
        self.test_frame = []
        for i in range(len(self.dt_results)):
            self.test_frame.append(self.dt_results[i]['frame_id'])

    def get_data_loader(self, distributed=False):
        return DataLoader(
            dataset=self,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            collate_fn=ActorNoiseCarlaDataset.collate_batch
        )

    def get_world_offset_and_scale(self, Town):
        offset = {'Town01': [-52.059906005859375, -52.04996085166931],
                'Town02': [-57.45972919464111, 55.3907470703125],
                'Town03': [-199.0638427734375, -259.27125549316406],
                'Town04': [-565.26904296875, -446.1461181640625], 
                'Town05': [-326.0445251464844, -257.8750915527344]
        }
        scale = {'Town01': 5,
                'Town02': 5,
                'Town03': 5,
                'Town04': 5,
                'Town05': 3,
        }
        return offset[Town], scale[Town]

    def get_points_in_a_rotated_box(self, corners, label_shape=[352, 400]):
        def minY(x0, y0, x1, y1, x):
            if x0 == x1:
                # vertical line, y0 is lowest
                return int(math.floor(y0))

            m = (y1 - y0) / (x1 - x0)

            if m >= 0.0:
                # lowest point is at left edge of pixel column
                return int(math.floor(y0 + m * (x - x0)))
            else:
                # lowest point is at right edge of pixel column
                return int(math.floor(y0 + m * ((x + 1.0) - x0)))


        def maxY(x0, y0, x1, y1, x):
            if x0 == x1:
                # vertical line, y1 is highest
                return int(math.ceil(y1))

            m = (y1 - y0) / (x1 - x0)

            if m >= 0.0:
                # highest point is at right edge of pixel column
                return int(math.ceil(y0 + m * ((x + 1.0) - x0)))
            else:
                # highest point is at left edge of pixel column
                return int(math.ceil(y0 + m * (x - x0)))


        # view_bl, view_tl, view_tr, view_br are the corners of the rectangle
        view = [(corners[i, 0], corners[i, 1]) for i in range(4)]

        pixels = []

        # find l,r,t,b,m1,m2
        l, m1, m2, r = sorted(view, key=lambda p: (p[0], p[1]))
        b, t = sorted([m1, m2], key=lambda p: (p[1], p[0]))

        lx, ly = l
        rx, ry = r
        bx, by = b
        tx, ty = t
        m1x, m1y = m1
        m2x, m2y = m2

        xmin = 0
        ymin = 0
        xmax = label_shape[1]
        ymax = label_shape[0]

        # inward-rounded integer bounds
        # note that we're clamping the area of interest to (xmin,ymin)-(xmax,ymax)
        lxi = max(int(math.ceil(lx)), xmin)
        rxi = min(int(math.floor(rx)), xmax)
        byi = max(int(math.ceil(by)), ymin)
        tyi = min(int(math.floor(ty)), ymax)

        x1 = lxi
        x2 = rxi

        for x in range(x1, x2):
            xf = float(x)

            if xf < m1x:
                # Phase I: left to top and bottom
                y1 = minY(lx, ly, bx, by, xf)
                y2 = maxY(lx, ly, tx, ty, xf)

            elif xf < m2x:
                if m1y < m2y:
                    # Phase IIa: left/bottom --> top/right
                    y1 = minY(bx, by, rx, ry, xf)
                    y2 = maxY(lx, ly, tx, ty, xf)

                else:
                    # Phase IIb: left/top --> bottom/right
                    y1 = minY(lx, ly, bx, by, xf)
                    y2 = maxY(tx, ty, rx, ry, xf)

            else:
                # Phase III: bottom/top --> right
                y1 = minY(bx, by, rx, ry, xf)
                y2 = maxY(tx, ty, rx, ry, xf)

            y1 = max(y1, byi)
            y2 = min(y2, tyi)

            for y in range(y1, y2):
                pixels.append((x, y))

        return pixels

    def update_label_map(self, map, bev_corners, reg_target):
        label_corners = self.transform_metric2label(bev_corners, ratio=self._geometry['ratio'], \
                                                    base_height=self._geometry['label_shape'][0],\
                                                    base_width=self._geometry['label_shape'][1] / 2)

        points = self.get_points_in_a_rotated_box(label_corners, self._geometry['label_shape'])

        for p in points:
            label_x = min(p[0], self._geometry['label_shape'][0] - 1)
            label_y = min(p[1], self._geometry['label_shape'][1] - 1)
            metric_x, metric_y = self.trasform_label2metric(np.array(p), ratio=self._geometry['ratio'], \
                                                            base_height=self._geometry['label_shape'][0],\
                                                            base_width=self._geometry['label_shape'][1] / 2)
            actual_reg_target = np.copy(reg_target)
            actual_reg_target[2] = reg_target[2] - metric_x
            actual_reg_target[3] = reg_target[3] - metric_y
            for i in range(4, len(reg_target)):
                actual_reg_target[i] = np.log(reg_target[i]) 

            map[label_x, label_y, 0] = 1.0
            if self._distribution_setting == True:
                map[label_x, label_y, 1:12] = actual_reg_target
            else:
                map[label_x, label_y, 1:7] = actual_reg_target

    def standardize(self, arr, mean, std):
        return (np.array(arr) - mean) / std

    def unstandardize(self, norm_arr, mean, std):
        return norm_arr * std + mean

    def get_corners(self, bbox, use_distribution):
        if  use_distribution == True:
            x, y, l, w, yaw, x_var, y_var, l_var, w_var, yaw_var = bbox
            reg_target = [np.cos(yaw), np.sin(yaw), x, y, w, l, yaw_var, x_var, y_var, w_var, l_var]
        else:
            x, y, l, w, yaw = bbox 
            reg_target = [np.cos(yaw), np.sin(yaw), x, y, w, l]
        
        bev_corners = np.zeros((4, 2), dtype=np.float32)
        # rear left
        bev_corners[0, 0] = x - l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[0, 1] = y - l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        # rear right
        bev_corners[1, 0] = x - l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[1, 1] = y - l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front right
        bev_corners[2, 0] = x + l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[2, 1] = y + l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front left
        bev_corners[3, 0] = x + l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[3, 1] = y + l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        return bev_corners, reg_target

    def update_label_map(self, map, bev_corners, reg_target):
        label_corners = self.transform_metric2label(bev_corners, ratio=self._geometry['ratio'], \
                                                    base_height=self._geometry['label_shape'][0],\
                                                    base_width=self._geometry['label_shape'][1] / 2)

        points = self.get_points_in_a_rotated_box(label_corners, self._geometry['label_shape'])

        for p in points:
            label_x = min(p[0], self._geometry['label_shape'][0] - 1)
            label_y = min(p[1], self._geometry['label_shape'][1] - 1)
            metric_x, metric_y = self.trasform_label2metric(np.array(p), ratio=self._geometry['ratio'], \
                                                            base_height=self._geometry['label_shape'][0],\
                                                            base_width=self._geometry['label_shape'][1] / 2)
            actual_reg_target = np.copy(reg_target)
            actual_reg_target[2] = reg_target[2] - metric_x
            actual_reg_target[3] = reg_target[3] - metric_y
            for i in range(4, len(reg_target)):
                actual_reg_target[i] = np.log(reg_target[i]) 

            map[label_x, label_y, 0] = 1.0
            if self._distribution_setting == True:
                map[label_x, label_y, 1:12] = actual_reg_target
            else:
                map[label_x, label_y, 1:7] = actual_reg_target

    def transform_gobal2metric(self, global_points, ego_pos, ego_yaw):
        trans_points = global_points - ego_pos
        label_x = trans_points[0] * np.cos(ego_yaw) + trans_points[1] * np.sin(ego_yaw)
        label_y = - trans_points[0] * np.sin(ego_yaw) + trans_points[1] * np.cos(ego_yaw)

        return [label_x, label_y]

    def transform_metric2label(self, metric, ratio=0.2, base_height=352, base_width=200):
        '''
        :param label: numpy array of shape [..., 2] of coordinates in metric space
        :return: numpy array of shape [..., 2] of the same coordinates in label_map space
        '''
        label = - metric / ratio
        label[:, 0] += base_height
        label[:, 1] += base_width

        return label

    def trasform_label2metric(self, label, ratio=0.2, base_height=352, base_width=200):
        '''
        :param label: numpy array of shape [..., 2] of coordinates in label map space
        :return: numpy array of shape [..., 2] of the same coordinates in metric space (lidar space)
        '''
        metric = np.copy(label)
        metric[0] -= base_height
        metric[1] -= base_width
        metric = - metric * ratio

        return metric
    
    def get_egopose_calib_token(self, sample_token):
        sample = self._nuscenes.get('sample', sample_token)
        lidar_top_data = self._nuscenes.get('sample_data', sample['data']['LIDAR_TOP']) 
        calib_token = lidar_top_data['calibrated_sensor_token']
        ego_pose_token = lidar_top_data['ego_pose_token']

        return ego_pose_token, calib_token

    def sample_occupancy_and_occlusion(self, sample_token):
        pic_height, pic_width = self._geometry['label_shape'][:2]
        ratio = self._geometry['ratio']
        img_cam_pos = [pic_height, pic_width/2]
        sample_occupancy = np.zeros([pic_height, pic_width]).astype('int32')
        sample_occlusion = np.ones([pic_height, pic_width]).astype('int32')

        frame_id = self._gt[idx]['frame_id']
        label_file = open(os.path.join(self.label_path, "%s.txt" %frame_id)) 

        for line in label_file.readlines():  
            attributes = line.split(" ")   
            if attributes[0] not in ['Car', 'Pedestrian', 'Cyclist', 'TrafficLight', 'TrafficSigns']:
                continue
    
            ############################
            # First stage: get occupancy
            ############################
            cam_x, cam_y, cam_z = attributes[11:14]
            lidar_x = float(cam_z)
            lidar_y = - float(cam_x)
            lidar_l = float(attributes[9])
            lidar_w = float(attributes[10])
            theta = float(attributes[14])

            # 调整坐标到图片坐标系
            img_x, img_y = self.transform_metric2label(np.array([[lidar_x, lidar_y]]))[0].astype('int32')
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
                        sample_occupancy[pix_x, pix_y] = 1

            ############################                
            # Second stage: get occlusion
            ############################
            corners, _ = self.get_corners([lidar_x, lidar_y, lidar_l, lidar_w, theta], use_distribution=False)
            label_corners = self.transform_metric2label(corners)

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
                sample_occlusion[list_of_points_indices] = 0
                
        return sample_occupancy, sample_occlusion

    def get_occupancy_and_occlusion(self, idx):
        occupancy = []
        occlusion = []

        for i in range(self._sweeps_len):   # 多帧代码无法适配carla数据集，因为是间隔着进行的val的
            sample_occupancy, sample_occlusion = self.sample_occupancy_and_occlusion(idx)
            occupancy.append(sample_occupancy)
            occlusion.append(sample_occlusion)

            if idx > 0:
                idx -= 1
            else:   # 为第一帧时，重复第一帧的数据
                pass
        
        # 将当前帧元素放在最顶层
        occupancy.reverse()
        occlusion.reverse()

        occupancy = np.stack(occupancy, axis=-1)      
        occlusion = np.stack(occlusion, axis=-1)

        return occupancy, occlusion

    def process_sample(self, idx):
        frame_id = self._gt[idx]['frame_id']
        label_file = open(os.path.join(self.label_path, "%s.txt" %frame_id))
        
        # 用于匹配GT和DT
        all_dt_data = self._gt[idx]['results']
        all_dt_match_token = []
        for dt_data in all_dt_data:
            all_dt_match_token.append(dt_data["detection"]['match_gt'])

        # 自车位姿，用于坐标系变换
        ego_pose = np.load(self.pose_path + '%s.npy' %frame_id)
        position = ego_pose[0:2]
        yaw = math.radians(ego_pose[4])

        GT_bbox = []
        label = []
        for gt_idx, line in enumerate(label_file.readlines()):  
            attributes = line.split(" ")   
            if attributes[0] not in ['Car', 'Pedestrian', 'Cyclist', 'TrafficLight', 'TrafficSigns']:
                continue

            cam_x, cam_y, cam_z = attributes[11:14]
            gt_x = float(cam_z)
            gt_y = - float(cam_x)
            gt_l = float(attributes[9])
            gt_w = float(attributes[10])
            gt_theta = float(attributes[14])

            if gt_x < self._geometry['W1'] or gt_x > self._geometry['W2'] or gt_y < self._geometry['L1'] or gt_y > self._geometry['L2'] :
                continue
            
            # 获的data和label
            if gt_idx in all_dt_match_token:
                dt_idx = all_dt_match_token.index(gt_idx)
                
                # get detection results
                det_data = all_dt_data[dt_idx]["detection"]
                
                dt_x = det_data['location'][2]
                dt_y = - det_data['location'][0]
                dt_l = det_data['dimensions'][2]
                dt_w = det_data['dimensions'][1]
                dt_theta = det_data['rotation_y']

                # get prediction results
                pred_data = all_dt_data[dt_idx]["prediction"]
                history_waypoints = pred_data['history_waypoints'][-1]
                center = self.transform_gobal2metric(history_waypoints, position, yaw)
                hist_x , hist_y = center[0], - center[1]

                global_waypoints = pred_data['future_waypoints']
                lidar_waypoints = []
                lidar_waypoints_st = [] # 存储标准化结果
                for i in range(global_waypoints.shape[0]):
                    center = self.transform_gobal2metric(global_waypoints[i], position, yaw)
                    lidar_waypoints.extend([center[0], - center[1]])
                    
                    # 在lidar视图上对轨迹点进行标准化，以当前位置为mean
                    if det_data['name'] == 'Car':         # 针对car类别，std选40
                        lidar_waypoints_st.extend([(center[0] - dt_x)/self._car_std, (- center[1] - dt_y)/self._car_std])
                    else:                                           # 针对pedestrian类别，std选1
                        lidar_waypoints_st.extend([(center[0] - dt_x)/self._pedestrian_std, (- center[1] - dt_y)/self._pedestrian_std])
                temp_label = [1, dt_x-gt_x, dt_y-gt_y, dt_l-gt_l, dt_w-gt_w, dt_theta-gt_theta]
                temp_label.extend(lidar_waypoints_st)
                label.append(temp_label)
                GT_bbox.append([hist_x , hist_y, gt_x, gt_y, gt_l, gt_w, gt_theta])
            else:
                label.append(18 * [0])
                GT_bbox.append([gt_x, gt_y, gt_x, gt_y, gt_l, gt_w, gt_theta])

        if len(GT_bbox) != 0:
            GT_bbox = np.stack(GT_bbox, axis=0)
            label = np.stack(label, axis=0)
        else:
            GT_bbox = np.array(GT_bbox)
            label = np.array(label)
        return GT_bbox, label

    def get_label(self, idx):
        label_map = np.zeros((self._geometry['label_shape'][0], self._geometry['label_shape'][1], 7), dtype=np.float32)
        label_list = []
        bev_bbox = []
        all_future_waypoints = []
        all_future_waypoints_st = []

        # 自车位姿，用于将gobal waypoint转到雷达坐标系
        frame_id = self._gt[idx]['frame_id']
        ego_pose = np.load(self.pose_path + '%s.npy' %frame_id)
        position = ego_pose[0:2]
        yaw = math.radians(ego_pose[4])

        all_gt_data = self._gt[idx]['results']
        for gt_data in all_gt_data:
            det_data = gt_data["detection"]
            pred_data = gt_data["prediction"]
            if det_data['name'] != 'Car': 
                continue

            # Step 1: 获取lidar坐标系下的检测结果，分为训练所用的label_map和测试所用label_list
            x = det_data['location'][2]
            y = - det_data['location'][0]
            l = det_data['dimensions'][2]
            w = det_data['dimensions'][1]
            theta = det_data['rotation_y']

            # Step 2:获取雷达坐标系下的预测结果
            global_waypoints = pred_data['future_waypoints']
            lidar_waypoints = []
            lidar_waypoints_st = [] # 存储标准化结果
            for i in range(global_waypoints.shape[0]):
                center = self.transform_gobal2metric(global_waypoints[i], position, yaw)
                lidar_waypoints.append([center[0], - center[1]])
                
                # 在lidar视图上对轨迹点进行标准化，以当前位置为mean
                if det_data['name'] == 'Car':         # 针对car类别，std选40
                    lidar_waypoints_st.append([(center[0] - x)/self._car_std, (- center[1] - y)/self._car_std])
                else:                                           # 针对pedestrian类别，std选1
                    lidar_waypoints_st.append([(center[0] - x)/self._pedestrian_std, (- center[1] - y)/self._pedestrian_std])
            lidar_waypoints = np.array(lidar_waypoints)
            lidar_waypoints_st = np.array(lidar_waypoints_st)

            # 进行筛选
            if x < self._geometry['W1'] or x > self._geometry['W2'] or y < self._geometry['L1'] or y > self._geometry['L2'] :
                continue    # 由于在目前检测结果是在范围 [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]下进行的，因此需要筛选
            if abs(lidar_waypoints[1][0] - lidar_waypoints[0][0]) > 10 or abs(lidar_waypoints[1][1] - lidar_waypoints[0][1]) > 10:
                continue    # 速度在x或y分量上超过20m/s(10m/0.5s)则被筛掉

            # 整理获取label
            corners, reg_target = self.get_corners([x, y, l, w, theta], use_distribution=False)
            self.update_label_map(label_map, corners, reg_target)
            label_list.append(corners)

            car_mean = np.array([17.45, -6.92, 4.72, 1.93, -1.11])
            car_std = np.array([10.43, 8.26, 0.26, 0.077, 1.61])
            norm_bev_bbox_para = self.standardize([x, y, l, w, theta], car_mean, car_std)
            bev_bbox.extend(norm_bev_bbox_para)

            all_future_waypoints.append(lidar_waypoints)
            all_future_waypoints_st.append(lidar_waypoints_st)

        if len(all_future_waypoints) != 0:
            future_waypoints = np.stack(all_future_waypoints, axis=0)
            future_waypoints_st = np.stack(all_future_waypoints_st, axis=0)
        else:
            future_waypoints = np.array([])
            future_waypoints_st = np.array([])

        assert len(label_list) == future_waypoints.shape[0]         # label_list应该和future_waypoints一一对应

        return label_map, label_list, bev_bbox, future_waypoints, future_waypoints_st

    def get_only_detection_label(self, idx):
        label_map = np.zeros((self._geometry['label_shape'][0], self._geometry['label_shape'][1], 7), dtype=np.float32)
        label_list = []

        frame_id = self._gt[idx]['frame_id']
        dt_result_idx = self.test_frame.index(frame_id)

        for det_data in self.dt_results[dt_result_idx]['dt']:
            if det_data['name'] != 'Car': 
                continue

            x = det_data['location'][2]
            y = - det_data['location'][0]
            l = det_data['dimensions'][2]
            w = det_data['dimensions'][1]
            theta = det_data['rotation_y']

            # 进行筛选
            if x < self._geometry['W1'] or x > self._geometry['W2'] or y < self._geometry['L1'] or y > self._geometry['L2'] :
                continue    # 由于在目前检测结果是在范围 [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]下进行的，因此需要筛选

            # 整理获取label
            corners, reg_target = self.get_corners([x, y, l, w, theta], use_distribution=False)
            self.update_label_map(label_map, corners, reg_target)
            label_list.append(corners)

        return label_map, label_list

    def __getitem__(self, index):
        assert index <= self.__len__()

        self.world_offset, self.world_scale = self.get_world_offset_and_scale(self._gt[index]['Town'])       # carla不同map由不同offset和scale

        # occupancy, occlusion = self.get_occupancy_and_occlusion(index)       # 实际的推理过程中，使用该方法\
        # occupancy = self._preprocess_data[index]['occupancy']
        # occlusion = self._preprocess_data[index]['occlusion']

        GT_bbox, label = self.process_sample(index)     # label: num of GT_bbox * 6(dectect or not, errx, erry, errw, errh errtheta)

        data_dict = {
            # 'occupancy': occupancy,
            # 'occlusion': occlusion,
            'GT_bbox': GT_bbox,
            'label': label
        }

        return data_dict

    def __len__(self):
        return len(self._gt)

    @staticmethod
    def load_data_to_gpu(batch_dict):
        for key, val in batch_dict.items():
            if not isinstance(val, np.ndarray):
                continue
            else:
                batch_dict[key] = torch.from_numpy(val).float().cuda()

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['occupancy', 'occlusion']:
                    values = []
                    for value in val:
                        values.append(value)
                    ret[key] = np.stack(values, axis=0)
                elif key in ['GT_bbox', 'label']:
                    values = []
                    for value in val:
                        if value.ndim == 1:
                            continue
                        values.append(value)
                    ret[key] = np.concatenate(values, axis=0)
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret
