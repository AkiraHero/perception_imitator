import pickle
import numpy as np
from torch.utils.data import DataLoader
import torch
import os
import math
import cv2
import sys
sys.path.append(os.getcwd())

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

def get_egopose_calib_token(sample_token, nuscenes):
    sample = nuscenes.get('sample', sample_token)
    lidar_top_data = nuscenes.get('sample_data', sample['data']['LIDAR_TOP']) 
    calib_token = lidar_top_data['calibrated_sensor_token']
    ego_pose_token = lidar_top_data['ego_pose_token']

    return ego_pose_token, calib_token

def transform_gobal2metric(ann_translation, ann_rotation, calib_data, ego_data):
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

nuscenes_type = 'v1.0-trainval'

nuscenes = NuScenes(nuscenes_type, dataroot="D:/PJLAB_Experiment/Data/nuScenes", verbose=False)
helper = PredictHelper(nuscenes)

gt_file_name = "mini_sim_model_gt.pkl" if nuscenes_type == "v1.0-mini" else "sim_model_gt.pkl"
gt_file = os.path.join('C:/Users/Sunyyyy/Desktop/Study/PJLAB/Code/ADModel_Pro/data', gt_file_name)
with open(gt_file, 'rb') as f:
    gt = pickle.load(f)   # 加载target detection model和target prediction model下的真值

all_bev_box = []

all_sample_token = list(gt.keys())
for sample_token in all_sample_token:
    ego_pose_token, calib_token= get_egopose_calib_token(sample_token, nuscenes)

    all_gt_data = gt[sample_token]
    calib_data = nuscenes.get('calibrated_sensor', calib_token)        
    ego_data = nuscenes.get('ego_pose', ego_pose_token)
    
    for gt_data in all_gt_data:
        det_data = gt_data["detection"]
        pred_data = gt_data["prediction"]
        if det_data['detection_name'] != 'car': 
            continue
        if det_data['detection_score'] >= 0.1:
            # Step 1: 获取lidar坐标系下的检测结果，分为训练所用的label_map和测试所用label_list
            center, yaw = transform_gobal2metric(det_data['translation'], det_data['rotation'], calib_data, ego_data)

            x = center[1]
            y = - center[0]
            l = np.array(det_data['size'])[1]
            w = np.array(det_data['size'])[0]
            theta = yaw

            # 进行筛选
            if x < 0 or x > 70.4 or y < -40 or y > 40:
                continue    # 由于在目前检测结果是在范围 [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]下进行的，因此需要筛选

            all_bev_box.append([x,y,l,w,theta])

all_bev_box = np.array(all_bev_box)
print(all_bev_box.shape)

car_mean = np.mean(all_bev_box, axis=0)
car_std = np.std(all_bev_box, axis=0)

print(car_mean)
print(car_std)