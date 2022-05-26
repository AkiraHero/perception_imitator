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

        gt_file_name = "mini_sim_model_gt.pkl" if self._nuscenes_type == "v1.0-mini" else "sim_model_gt.pkl"
        gt_file = os.path.join(self._data_root, gt_file_name)
        with open(gt_file, 'rb') as f:
            self._gt = pickle.load(f)   # 加载target detection model和target prediction model下的真值

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

    def transform_gobal2metric(self, ann_translation, ann_rotation, calib_data, ego_data):
        # global frame
        center = np.array(ann_translation)
        orientation = np.array(ann_rotation)
        # 从global frame转换到ego vehicle frame
        quaternion = Quaternion(ego_data['rotation']).inverse
        center -= np.array(ego_data['translation'])
        center = np.dot(quaternion.rotation_matrix, center)
        orientation = quaternion * orientation
        # 从ego vehicle frame转换到sensor frame
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

        # 对sample中的每个annotation进行坐标变换(global -> lidar)
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

        sample_token = list(self._gt.keys())[idx]
        sample = self._nuscenes.get('sample', sample_token)
        for i in range(self._sweeps_len):
            sample_occupancy, sample_occlusion = self.sample_occupancy_and_occlusion(sample_token)
            occupancy.append(sample_occupancy)
            occlusion.append(sample_occlusion)

            if sample['prev'] in self._gt.keys():
                sample_token = sample['prev']
                sample = self._nuscenes.get('sample', sample_token)
            else:   # 当真值中没有prev_sample时，则重复当前帧结果
                pass
        
        # 将当前帧元素放在最顶层
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
        # 对sample中的每个annotation进行坐标变换(global -> lidar)
        for annotation in sample_annotation:
            anno_token = annotation['token']
            ann = self._nuscenes.get('sample_annotation', anno_token)
            
            # 只针对vehicle类别
            if 'vehicle' not in ann['category_name']:
                continue
            
            # 不在range内的剔除
            center, yaw = self.transform_gobal2metric(ann['translation'], ann['rotation'], calib_data, ego_data)
            gt_x = center[1]
            gt_y = - center[0]
            gt_l = np.array(ann['size'])[1]
            gt_w = np.array(ann['size'])[0]
            gt_theta = yaw
            if gt_x < self._geometry['W1'] or gt_x > self._geometry['W2'] or gt_y < self._geometry['L1'] or gt_y > self._geometry['L2'] :
                continue
            
            # 获的data和label
            GT_bbox.append([gt_x, gt_y, gt_l, gt_w, gt_theta])
            if anno_token in all_dt_match_token:
                dt_idx = all_dt_match_token.index(anno_token)

                det_data = all_dt_data[dt_idx]["detection"]
                center, yaw = self.transform_gobal2metric(det_data['translation'], det_data['rotation'], calib_data, ego_data)
                dt_x = center[1]
                dt_y = - center[0]
                dt_l = np.array(det_data['size'])[1]
                dt_w = np.array(det_data['size'])[0]
                dt_theta = yaw

                label.append([1, dt_x-gt_x, dt_y-gt_y, dt_l-gt_l, dt_w-gt_w, dt_theta-gt_theta]) 
            else:
                label.append([0, 0, 0, 0, 0, 0])

        GT_bbox = np.stack(GT_bbox, axis=0)
        label = np.stack(label, axis=0)

        return GT_bbox, label

    def __getitem__(self, index):
        assert index <= self.__len__()

        occupancy, occlusion = self.get_occupancy_and_occlusion(index)       # 实际的推理过程中，使用该方法
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
                        values.append(value)
                    ret[key] = np.concatenate(values, axis=0)
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret
