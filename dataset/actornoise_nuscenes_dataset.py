import os
import math
import cv2
import random
from cProfile import label
from cmath import pi
from optparse import Values
import pickle
from matplotlib.style import use
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.dataset_base import DatasetBase
from utils.nuscenes.nuscenes import NuScenes
from utils.nuscenes.eval.prediction.splits import get_prediction_challenge_split
from utils.nuscenes.prediction import PredictHelper
from utils.nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from utils.nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from utils.nuscenes.prediction.input_representation.interface import InputRepresentation
from utils.nuscenes.prediction.input_representation.combinators import Rasterizer
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

class ActorNoiseNuscenesDataset(DatasetBase):
    
    def __init__(self, config):
        super(DatasetBase, self).__init__()
        self._is_train = config['paras']['for_train']
        self._batch_size = config['paras']['batch_size']
        self._data_root = config['paras']['data_root']
        self._num_workers = config['paras']['num_workers']
        self._shuffle = config['paras']['shuffle']
        self._nuscenes_type = config['paras']['nuscenes_type']
        self._geometry = config['paras']['geometry']
        self._sweeps_len = config['paras']['sweeps_len']
        assert self._sweeps_len > 0 and isinstance(self._sweeps_len, int)
        self._distribution_setting = config['paras']['FP_distribution']
        self._car_std = config['paras']['waypoints_std']['car']
        self._pedestrian_std = config['paras']['waypoints_std']['pedestrian']

        self._nuscenes = NuScenes(self._nuscenes_type, dataroot="E:/PJLAB_Experiment/Data/nuScenes", verbose=False)
        self._helper = PredictHelper(self._nuscenes)

        if self._is_train == True:
            gt_file_name = "mini_sim_model_gt.pkl" if self._nuscenes_type == "v1.0-mini" else "pointpillar_sim_model_gt.pkl"
        else:
            gt_file_name = "mini_sim_model_gt_test.pkl" if self._nuscenes_type == "v1.0-mini" else "centerpoint_sim_model_gt_test.pkl"
        gt_file = os.path.join(self._data_root, gt_file_name)
        with open(gt_file, 'rb') as f:
            self._gt = pickle.load(f)   # ??????target detection model???target prediction model????????????

        # ??????????????????????????????????????????
        with open(os.path.join(self._data_root, "nuscenes_preprocess_data.pkl"), 'rb') as f:
            self._preprocess_data = pickle.load(f) 

    def get_data_loader(self, distributed=False):
        return DataLoader(
            dataset=self,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            collate_fn=ActorNoiseNuscenesDataset.collate_batch
        )

    def label_str2num(self, clss):
        d = {
            'Car': 1,
            'Van': 1,
            'Truck': 1,
            'Tram': 1,
            'Misc': 1,
            'Pedestrian': 2,
            'Person_sitting':2,
            'Cyclist': 3
        }
        return d[clss]
    
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

    def transform_gobal2metric(self, ann_translation, ann_rotation, calib_data, ego_data):
        # global frame
        center = np.array(ann_translation)
        orientation = np.array(ann_rotation)
        # ???global frame?????????ego vehicle frame
        quaternion = Quaternion(ego_data['rotation']).inverse
        center -= np.array(ego_data['translation'])
        center = np.dot(quaternion.rotation_matrix, center)
        orientation = quaternion * orientation
        # ???ego vehicle frame?????????sensor frame
        quaternion = Quaternion(calib_data['rotation']).inverse
        center -= np.array(calib_data['translation'])
        center = np.dot(quaternion.rotation_matrix, center)
        orientation = quaternion * orientation
        v = np.dot(orientation.rotation_matrix, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0]) - math.pi / 2

        return center, yaw

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
        ego_pose_token, calib_token  = self.get_egopose_calib_token(sample_token)

        pic_height, pic_width = self._geometry['label_shape'][:2]
        ratio = self._geometry['ratio']
        img_cam_pos = [pic_height, pic_width/2]
        sample_occupancy = np.zeros([pic_height, pic_width]).astype('int32')
        sample_occlusion = np.ones([pic_height, pic_width]).astype('int32')

        sample_annotation = self._helper.get_annotations_for_sample(sample_token)
        calib_data = self._nuscenes.get('calibrated_sensor', calib_token)
        ego_data = self._nuscenes.get('ego_pose', ego_pose_token)

        # ???sample????????????annotation??????????????????(global -> lidar)
        for annotation in sample_annotation:
            anno_token = annotation['token']
            ann = self._nuscenes.get('sample_annotation', anno_token)
            center, yaw = self.transform_gobal2metric(ann['translation'], ann['rotation'], calib_data, ego_data)
    
            ############################
            # First stage: get occupancy
            ############################
            lidar_x = center[1]
            lidar_y = - center[0]
            lidar_l = np.array(ann['size'])[1]
            lidar_w = np.array(ann['size'])[0]
            theta = yaw

            # ??????????????????????????????
            img_x, img_y = self.transform_metric2label(np.array([[lidar_x, lidar_y]]))[0].astype('int32')
            img_l = lidar_l / ratio
            img_w = lidar_w / ratio

            # get occupancy
            # ?????????????????????????????????
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
                # ??????????????????bev???????????????
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

        sample_token = list(self._gt.keys())[idx]
        sample = self._nuscenes.get('sample', sample_token)
        for i in range(self._sweeps_len):
            sample_occupancy, sample_occlusion = self.sample_occupancy_and_occlusion(sample_token)
            occupancy.append(sample_occupancy)
            occlusion.append(sample_occlusion)

            if sample['prev'] in self._gt.keys():
                sample_token = sample['prev']
                sample = self._nuscenes.get('sample', sample_token)
            else:   # ??????????????????prev_sample??????????????????????????????
                pass
        
        # ?????????????????????????????????
        occupancy.reverse()
        occlusion.reverse()

        occupancy = np.stack(occupancy, axis=-1)      
        occlusion = np.stack(occlusion, axis=-1)

        return occupancy, occlusion

    def process_sample(self, idx):
        sample_token = list(self._gt.keys())[idx]
        ego_pose_token, calib_token  = self.get_egopose_calib_token(sample_token)

        sample_annotation = self._helper.get_annotations_for_sample(sample_token)
        calib_data = self._nuscenes.get('calibrated_sensor', calib_token)
        ego_data = self._nuscenes.get('ego_pose', ego_pose_token)

        all_dt_data = self._gt[sample_token]
        all_dt_match_token = []
        for dt_data in all_dt_data:
            all_dt_match_token.append(dt_data["detection"]['match_gt'])

        GT_bbox = []
        label = []
        # ???sample????????????annotation??????????????????(global -> lidar)
        for annotation in sample_annotation:
            anno_token = annotation['token']
            ann = self._nuscenes.get('sample_annotation', anno_token)
            
            # ?????????vehicle??????
            if 'vehicle' not in ann['category_name']:
                continue
            
            # ??????range????????????
            center, yaw = self.transform_gobal2metric(ann['translation'], ann['rotation'], calib_data, ego_data)
            gt_x = center[1]
            gt_y = - center[0]
            gt_l = np.array(ann['size'])[1]
            gt_w = np.array(ann['size'])[0]
            gt_theta = yaw
            if gt_x < self._geometry['W1'] or gt_x > self._geometry['W2'] or gt_y < self._geometry['L1'] or gt_y > self._geometry['L2'] :
                continue
            
            # ??????data???label
            GT_bbox.append([gt_x, gt_y, gt_l, gt_w, gt_theta])
            if anno_token in all_dt_match_token:
                dt_idx = all_dt_match_token.index(anno_token)
                
                # get detection results
                det_data = all_dt_data[dt_idx]["detection"]
                center, yaw = self.transform_gobal2metric(det_data['translation'], det_data['rotation'], calib_data, ego_data)
                dt_x = center[1]
                dt_y = - center[0]
                dt_l = np.array(det_data['size'])[1]
                dt_w = np.array(det_data['size'])[0]
                dt_theta = yaw

                # get prediction results
                pred_data = all_dt_data[dt_idx]["prediction"]
                global_waypoints = pred_data['future_waypoints']
                global_waypoints = np.insert(global_waypoints, 2, values=0, axis=1)
                lidar_waypoints = []
                lidar_waypoints_st = [] # ?????????????????????
                for i in range(global_waypoints.shape[0]):
                    center, _ = self.transform_gobal2metric(global_waypoints[i], np.zeros(4), calib_data, ego_data)
                    lidar_waypoints.extend([center[1], - center[0]])
                    
                    # ???lidar?????????????????????????????????????????????????????????mean
                    if det_data['detection_name'] == 'car':         # ??????car?????????std???40
                        lidar_waypoints_st.extend([(center[1] - dt_x)/self._car_std, (- center[0] - dt_y)/self._car_std])
                    else:                                           # ??????pedestrian?????????std???1
                        lidar_waypoints_st.extend([(center[1] - dt_x)/self._pedestrian_std, (- center[0] - dt_y)/self._pedestrian_std])

                temp_label = [1, dt_x-gt_x, dt_y-gt_y, dt_l-gt_l, dt_w-gt_w, dt_theta-gt_theta]
                temp_label.extend(lidar_waypoints_st)
                label.append(temp_label)
            else:
                label.append(18 * [0])

        if len(GT_bbox) != 0:
            GT_bbox = np.stack(GT_bbox, axis=0)
            label = np.stack(label, axis=0)
        else:
            GT_bbox = np.array(GT_bbox)
            label = np.array(label)
        return GT_bbox, label

    def get_label(self, idx):
        
        if self._distribution_setting == True:
            label_map = np.zeros((self._geometry['label_shape'][0], self._geometry['label_shape'][1], 12), dtype=np.float32)
        else:
            label_map = np.zeros((self._geometry['label_shape'][0], self._geometry['label_shape'][1], 7), dtype=np.float32)
        label_list = []
        bev_bbox = []
        all_future_waypoints = []
        all_future_waypoints_st = []

        ### ?????????????????????target_dt & target_pred?????????###
        if self._distribution_setting == True:  
            fp_bboxes = self._fp_distribution['fp_mean'][idx]   # ??????????????????????????????????????????
            fp_var = self._fp_distribution['fp_var'][idx]
            fp_num = self._fp_distribution['fp_num'][idx]       # ?????????????????????392??????????????????????????????FP????????????5????????????FP
            scores = self._fp_distribution['fp_score'][idx]

            for i in range(len(fp_bboxes)):     # ????????????FP???
                if fp_num[i] >= 50 and scores[i] >= 0.3:
                    x = fp_bboxes[i][0]
                    y = fp_bboxes[i][1]
                    l = fp_bboxes[i][3]
                    w = fp_bboxes[i][4]
                    yaw = fp_bboxes[i][6]

                    x_var = fp_var[i][0]
                    y_var = fp_var[i][1]
                    l_var = fp_var[i][3]
                    w_var = fp_var[i][4]
                    yaw_var = fp_var[i][6]

                    corners, reg_target = self.get_corners([x, y, l, w, yaw, x_var, y_var, l_var, w_var, yaw_var], use_distribution=True)
                    self.update_label_map(label_map, corners, reg_target)
                    label_list.append(corners)
        #####################################################

        else:
            sample_token = list(self._gt.keys())[idx]
            ego_pose_token, calib_token= self.get_egopose_calib_token(sample_token)

            all_gt_data = self._gt[sample_token]
            calib_data = self._nuscenes.get('calibrated_sensor', calib_token)        
            ego_data = self._nuscenes.get('ego_pose', ego_pose_token)

            for gt_data in all_gt_data:
                det_data = gt_data["detection"]
                pred_data = gt_data["prediction"]
                if det_data['detection_name'] != 'car': 
                    continue
                if det_data['detection_score'] >= 0.05:
                    # Step 1: ??????lidar???????????????????????????????????????????????????label_map???????????????label_list
                    center, yaw = self.transform_gobal2metric(det_data['translation'], det_data['rotation'], calib_data, ego_data)

                    x = center[1]
                    y = - center[0]
                    l = np.array(det_data['size'])[1]
                    w = np.array(det_data['size'])[0]
                    theta = yaw

                    # Step 2:???????????????????????????????????????
                    global_waypoints = pred_data['future_waypoints']
                    global_waypoints = np.insert(global_waypoints, 2, values=0, axis=1)
                    lidar_waypoints = []
                    lidar_waypoints_st = [] # ?????????????????????
                    for i in range(global_waypoints.shape[0]):
                        center, _ = self.transform_gobal2metric(global_waypoints[i], np.zeros(4), calib_data, ego_data)
                        lidar_waypoints.append([center[1], - center[0]])
                        
                        # ???lidar?????????????????????????????????????????????????????????mean
                        if det_data['detection_name'] == 'car':         # ??????car?????????std???40
                            lidar_waypoints_st.append([(center[1] - x)/self._car_std, (- center[0] - y)/self._car_std])
                        else:                                           # ??????pedestrian?????????std???1
                            lidar_waypoints_st.append([(center[1] - x)/self._pedestrian_std, (- center[0] - y)/self._pedestrian_std])
                    lidar_waypoints = np.array(lidar_waypoints)
                    lidar_waypoints_st = np.array(lidar_waypoints_st)

                    # ????????????
                    if x < self._geometry['W1'] or x > self._geometry['W2'] or y < self._geometry['L1'] or y > self._geometry['L2'] :
                        continue    # ??????????????????????????????????????? [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]?????????????????????????????????
                    if abs(lidar_waypoints[1][0] - lidar_waypoints[0][0]) > 10 or abs(lidar_waypoints[1][1] - lidar_waypoints[0][1]) > 10:
                        continue    # ?????????x???y???????????????20m/s(10m/0.5s)????????????

                    # ????????????label
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

        assert len(label_list) == future_waypoints.shape[0]         # label_list?????????future_waypoints????????????

        return label_map, label_list, bev_bbox, future_waypoints, future_waypoints_st

    def __getitem__(self, index):
        assert index <= self.__len__()

        occupancy, occlusion = self.get_occupancy_and_occlusion(index)       # ??????????????????????????????????????????\
        # occupancy = self._preprocess_data[index]['occupancy']
        # occlusion = self._preprocess_data[index]['occlusion']

        GT_bbox, label = self.process_sample(index)     # label: num of GT_bbox * 6(dectect or not, errx, erry, errw, errh errtheta)

        data_dict = {
            'occupancy': occupancy,
            'occlusion': occlusion,
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
