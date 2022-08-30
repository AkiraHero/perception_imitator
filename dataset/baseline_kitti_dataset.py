import os
import math
import cv2
import random
from cProfile import label
from cmath import isnan, pi
from optparse import Values
import pickle
from matplotlib.style import use
from sklearn.mixture import GaussianMixture
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

class BaselineKittiDataset(DatasetBase):
    
    def __init__(self, config):
        super(DatasetBase, self).__init__()
        self._is_train = config['paras']['for_train']
        self._batch_size = config['paras']['batch_size']
        self._data_root = config['paras']['data_root']
        self._num_workers = config['paras']['num_workers']
        self._shuffle = config['paras']['shuffle']
        self._geometry = config['paras']['geometry']
        self._sweeps_len = config['paras']['sweeps_len']
        assert self._sweeps_len > 0 and isinstance(self._sweeps_len, int)
        self._target_model = config['paras']['target_model']
        self._is_interface = config['paras']['interface']
        self._distribution_setting = config['paras']['FP_distribution']

        # 获取检测结果
        detect_result_file = os.path.join("./data", "kitti_%s_match_gt.pkl" %self._target_model)
        with open(detect_result_file, 'rb') as f:     # 加载检测结果
            self.all_dt_results = pickle.load(f)
        random.seed(2022)
        random.shuffle(self.all_dt_results)
        offset = int(len(self.all_dt_results) * 0.8)
        if self._is_train == True:
            self.dt_results = self.all_dt_results[:offset]
        else:
            self.dt_results = self.all_dt_results[offset:]

        # 获取所有DT和GT x, y, logl, logw, cost, sint的误差数组，用于后续使用高斯混合分布进行逼近
        self.GMM = self.get_box_error_GMM(8)

    def get_box_error_GMM(self, components=8):
        err_x, err_y, err_log_l, err_log_w, err_cost, err_sint = [], [], [], [], [], []

        for frame in self.dt_results:
            all_dt_data = frame['dt']
            label = frame['gt']

            for det_data in all_dt_data:

                if det_data['name'] != 'Car': 
                    continue

                x = det_data['location'][2]
                y = - det_data['location'][0]
                log_l = np.log(det_data['dimensions'][0])
                log_w = np.log(det_data['dimensions'][2])
                cost = np.cos(det_data['rotation_y'])
                sint = np.sin(det_data['rotation_y'])
            
                # DT对应的GT
                if det_data['match_gt'] == -1:
                    continue
                GT = label[det_data['match_gt']]

                gt_x = GT['location'][2]
                gt_y = - GT['location'][0]
                gt_log_l = np.log(GT['dimensions'][0])
                gt_log_w = np.log(GT['dimensions'][2])
                gt_cost = np.cos(GT['rotation_y'])
                gt_sint = np.sin(GT['rotation_y'])

                err_x.extend([x - gt_x])
                err_y.extend([y - gt_y])
                err_log_l.extend([log_l - gt_log_l])
                err_log_w.extend([log_w - gt_log_w])
                err_cost.extend([cost - gt_cost])
                err_sint.extend([sint - gt_sint])

        GMM_x = GaussianMixture(n_components=components, covariance_type='full', random_state=np.random).fit(np.array(err_x).reshape(-1, 1))
        GMM_y = GaussianMixture(n_components=components, covariance_type='full', random_state=np.random).fit(np.array(err_y).reshape(-1, 1))
        GMM_log_l = GaussianMixture(n_components=components, covariance_type='full', random_state=np.random).fit(np.array(err_log_l).reshape(-1, 1))
        GMM_log_w = GaussianMixture(n_components=components, covariance_type='full', random_state=np.random).fit(np.array(err_log_w).reshape(-1, 1))
        GMM_cost = GaussianMixture(n_components=components, covariance_type='full', random_state=np.random).fit(np.array(err_cost).reshape(-1, 1))
        GMM_sint = GaussianMixture(n_components=components, covariance_type='full', random_state=np.random).fit(np.array(err_sint).reshape(-1, 1))

        return [GMM_x, GMM_y, GMM_log_l, GMM_log_w, GMM_cost, GMM_sint]

    def get_data_loader(self, distributed=False):
        return DataLoader(
            dataset=self,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            collate_fn=BaselineKittiDataset.collate_batch
        )

    def get_crop(self, img, ego_pix, meters_behind, meters_ahead, meters_left, meters_right, resolution):
        pix_ahead = meters_ahead * resolution
        pix_behind = meters_behind * resolution
        pix_left = meters_left * resolution
        pix_right = meters_right * resolution

        return img[slice(int(ego_pix[1]-pix_ahead), int(ego_pix[1]-pix_behind)), slice(int(ego_pix[0]+pix_left), int(ego_pix[0]+pix_right)), ...]

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

    def sample_occupancy_and_occlusion(self, idx):
        pic_height, pic_width = self._geometry['label_shape'][:2]
        ratio = self._geometry['ratio']
        img_cam_pos = [pic_height, pic_width/2]
        sample_occupancy = np.zeros([pic_height, pic_width]).astype('int32')
        sample_occlusion = np.ones([pic_height, pic_width]).astype('int32')

        frame = self.dt_results[idx]
        all_gt_data = frame['gt']

        for gt_data in all_gt_data:
            if gt_data['name'] == "DontCare":
                continue
    
            ############################
            # First stage: get occupancy
            ############################
            lidar_x = gt_data['location'][2]
            lidar_y = - gt_data['location'][0]
            lidar_l = gt_data['dimensions'][0]
            lidar_w = gt_data['dimensions'][2]
            theta = gt_data['rotation_y'] + np.pi / 2

            # 调整坐标到图片坐标系
            img_x, img_y = self.transform_metric2label(np.array([[lidar_x, lidar_y]]))[0].astype('int32')
            img_l = lidar_l / ratio
            img_w = lidar_w / ratio

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

    def get_occupancy(self, idx):
        return self._occ[idx]['occupancy']

    def get_occlusion(self, idx):
        return self._occ[idx]['occlusion']
    
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
    
    def get_only_detection_label(self, idx):
        label_map = np.zeros((self._geometry['label_shape'][0], self._geometry['label_shape'][1], 7), dtype=np.float32)
        label_list = []

        for det_data in self.dt_results[idx]['dt']:

            if det_data['name'] != 'Car': 
                continue

            # 获取lidar坐标系下的检测结果，分为训练所用的label_map和测试所用label_list
            x = det_data['location'][2]
            y = - det_data['location'][0]
            l = det_data['dimensions'][0]
            w = det_data['dimensions'][2]
            theta = det_data['rotation_y'] + np.pi / 2

            # 进行筛选
            if x < self._geometry['W1'] or x > self._geometry['W2'] or y < self._geometry['L1'] or y > self._geometry['L2'] :
                continue    # 由于在目前检测结果是在范围 [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]下进行的，因此需要筛选

            # 整理获取label
            corners, reg_target = self.get_corners([x, y, l, w, theta], use_distribution=False)
            self.update_label_map(label_map, corners, reg_target)
            label_list.append(corners)

        return label_map, label_list

    def get_gt_info(self, idx):
        gt_map = np.zeros((self._geometry['label_shape'][0], self._geometry['label_shape'][1], 7), dtype=np.float32)
        gt_list = []

        for gt_data in self.dt_results[idx]['gt']:
            if gt_data['name'] != 'Car': 
                continue

            x = gt_data['location'][2]
            y = - gt_data['location'][0]
            l = gt_data['dimensions'][0]
            w = gt_data['dimensions'][2]
            theta = gt_data['rotation_y'] + np.pi / 2

            # 整理获取label
            corners, reg_target = self.get_corners([x, y, l, w, theta], use_distribution=False)
            self.update_label_map(gt_map, corners, reg_target)
            gt_list.append(corners)

        return gt_map, gt_list
   
    def get_gaussian_noise(self, idx, mu=0, sigma=1 ,drop=0):
        gaussian_noise_list = []

        for gt_data in self.dt_results[idx]['gt']:  
            if gt_data['name'] != 'Car':    # 只针对Car类型增加噪声
                continue
            if np.random.choice([0, 1], p=[1 - drop, drop]):    # 按概率进行筛选模拟FN
                continue

            x = gt_data['location'][2]
            y = - gt_data['location'][0]
            log_l = np.log(gt_data['dimensions'][0])
            log_w = np.log(gt_data['dimensions'][2])
            cos_t, sin_t = np.cos(gt_data['rotation_y']), np.sin(gt_data['rotation_y'])

            # Add Gaussian Noise
            x += random.gauss(mu, sigma)
            y += random.gauss(mu, sigma)
            log_l += random.gauss(mu, sigma)
            log_w += random.gauss(mu, sigma)
            cos_t += random.gauss(mu, sigma)
            sin_t += random.gauss(mu, sigma)

            l = np.exp(log_l)
            w = np.exp(log_w)
            theta = math.atan2(sin_t, cos_t) + np.pi / 2

            corners, _ = self.get_corners([x, y, l, w, theta], use_distribution=False)
            gaussian_noise_list.append(corners)

        return gaussian_noise_list

    def get_multimodal_noise(self, idx, drop=0):
        # fig = plt.figure(figsize=(5, 1.7))
        # ax = fig.add_subplot(111)

        # x = np.linspace(-1, 1, 1000)
        # logprob = self.GMM[1].score_samples(x.reshape(-1, 1))
        # pdf = np.exp(logprob)
        # ax.plot(x, pdf, '-k')
        # plt.show()

        multimodal_noise_list = []

        for gt_data in self.dt_results[idx]['gt']:  
            if gt_data['name'] != 'Car':    # 只针对Car类型增加噪声
                continue
            if np.random.choice([0, 1], p=[1 - drop, drop]):    # 按概率进行筛选模拟FN
                continue

            x = gt_data['location'][2]
            y = - gt_data['location'][0]
            log_l = np.log(gt_data['dimensions'][0])
            log_w = np.log(gt_data['dimensions'][2])
            cos_t, sin_t = np.cos(gt_data['rotation_y']), np.sin(gt_data['rotation_y'])

            # Add Gaussian Noise
            x += self.GMM[0].sample(1)[0]
            y += self.GMM[1].sample(1)[0]
            log_l += self.GMM[2].sample(1)[0]
            log_w += self.GMM[3].sample(1)[0]
            cos_t += self.GMM[4].sample(1)[0]
            sin_t += self.GMM[5].sample(1)[0]

            l = np.exp(log_l)
            w = np.exp(log_w)
            theta = math.atan2(sin_t, cos_t) + np.pi / 2

            corners, _ = self.get_corners([x, y, l, w, theta], use_distribution=False)
            multimodal_noise_list.append(corners)

        return multimodal_noise_list

    def __getitem__(self, index):
        assert index <= self.__len__()
        occupancy, occlusion = self.get_occupancy_and_occlusion(index)       # 实际的推理过程中，使用该方法 

        label_map, label_list = self.get_only_detection_label(index)
        gt_map, gt_list = self.get_gt_info(index)

        data_dict = {
            'occupancy': occupancy,
            'occlusion': occlusion,
            'label_map': label_map,
            'label_list': label_list,
            'gt_map': gt_map,
            'gt_list': gt_list
        }

        return data_dict

    def __len__(self):
        return len(self.dt_results)

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
                if key in ['occupancy', 'occlusion', 'label_map', 'gt_map']:
                    values = []
                    for value in val:
                        values.append(value)
                    ret[key] = np.stack(values, axis=0)
                elif key in ['label_list', 'gt_list']:
                    values = []
                    for value in val:
                        values.append(value)
                    ret[key] = values
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret


    def get_yiming_need(self, idx, corner):
        frame_id = self.dt_results[idx]['frame_id']

        x = corner[:, 0].sum() / 4
        y = corner[:, 1].sum() / 4
        # center = np.array([-y, x, 0.7])

        yaw = np.arctan2(corner[3][0] - corner[0][0], -(corner[3][1] - corner[0][1]))
        # rotation = Quaternion(axis=[0, 0, 1], angle=yaw)

        # rotation = rotation.elements.tolist()
        # translation = center.tolist()

        l = two_points_distance(corner[0], corner[3])
        w = two_points_distance(corner[0], corner[1])
        # size = [w, l, 1.5]

        # return frame_id, translation, size, rotation
        return frame_id, x, y, l, w, yaw