import pickle
from matplotlib.style import use
import torch
import os
import math
import cv2
from dataset.dataset_base import DatasetBase
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
from utils.preprocess import two_points_2_line, two_points_distance, euclidean_distance

'''
Dataset for easy scene expression loading
    occupancy: H * W
    occlusion: H * W
    heatmap: H * W
    label_map: H * W * 7 , 7 channels are [cls_FP_or_not, cos(yaw), sin(yaw), x, y, log(w), log(l)]
'''

class FpSceneOccHeatmapDataset(DatasetBase):
    def __init__(self, config):
        super(DatasetBase, self).__init__()
        self._is_train = config['paras']['for_train']
        self._batch_size = config['paras']['batch_size']
        self._data_root = config['paras']['data_root']
        self._num_workers = config['paras']['num_workers']
        self._shuffle = config['paras']['shuffle']
        self._geometry = config['paras']['geometry']
        self._distribution_setting = config['paras']['FP_distribution']
        # 均值和方差
        self.target_mean = np.array([0.008, 0.001, 0.202, 0.2, 0.43, 1.368])
        self.target_std_dev = np.array([0.866, 0.5, 0.954, 0.668, 0.09, 0.111])
        
        # self._occ = []
        # for i in range(1):
        #     occ_file = os.path.join(self._data_root, "easy_scene_%d.pkl" %i)
        #     with open(occ_file, 'rb') as f:
        #         occ_split = pickle.load(f)
        #         self._occ.extend(occ_split)

        heatmap_file = os.path.join(self._data_root, "GTheatmap_aug.pkl")
        with open(heatmap_file, 'rb') as f:
            self._heatmap = pickle.load(f)
        
        exp_name = config['paras']['exp_name_root']
        matching_name = os.path.join(exp_name, 'gt_dt_matching_res.pkl')
        result_name = os.path.join(exp_name, 'result.pkl')
        with open(matching_name, 'rb') as f:
            self._gt_dt_matching_res = pickle.load(f)   # 存储了真值
        with open(result_name, 'rb') as f:
            self._result = pickle.load(f)   # self._gt_dt_matching_res和self._result可以获取一次实验中的FP数据

        if self._distribution_setting == True:
            fp_distribution_name = os.path.join(self._data_root, "fp_distribution.pkl")     # 存储的是所有实验得到FP的各个参数的均值和方差
            with open(fp_distribution_name, 'rb') as f:
                self._fp_distribution = pickle.load(f)   # 存储的是所有实验得到FP的各个参数的均值和方差

    def get_data_loader(self, distributed=False):
        return DataLoader(
            dataset=self,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            collate_fn=FpSceneOccHeatmapDataset.collate_batch
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

    def get_occupancy(self, idx):
        return self._occ[idx]['occupancy']

    def get_occlusion(self, idx):
        return self._occ[idx]['occlusion']
    
    def get_occupancy_and_occlusion(self, idx):
        gt_annos = self._gt_dt_matching_res['gt_annos']
        gt_bboxes = gt_annos[idx]['gt_boxes_lidar']    # 获取当前Lidar id下的gt检测框

        pic_height, pic_width = self._geometry['label_shape'][:2]
        ratio = self._geometry['ratio']

        img_cam_pos = [pic_height, pic_width/2]
        occupancy = np.zeros([pic_height, pic_width]).astype('int32')
        occlusion = np.ones([pic_height, pic_width]).astype('int32')

        for one_gtbbox in gt_bboxes:
            # 高长宽
            lidar_x = one_gtbbox[0]
            lidar_y = one_gtbbox[1]
            lidar_l = one_gtbbox[3]
            lidar_w = one_gtbbox[4]
            theta = one_gtbbox[6]

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
                        occupancy[pix_x, pix_y] = 1  

            # get occlusion
            corners, _ = self.get_corners([lidar_x, lidar_y, lidar_l, lidar_w, theta], use_distribution=False)
            label_corners = self.transform_metric2label(corners)

            k_list = []
            b_list = [] 
            dist_list = []
            intersection_points = []
            
            if (label_corners < 0).sum() > 0:
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

        return occupancy, occlusion

    def get_heatmap(self, idx):
        return self._heatmap[idx]['gt_heatmap'] / 255

    def get_label(self, idx):
        if self._distribution_setting == True:
            label_map = np.zeros((self._geometry['label_shape'][0], self._geometry['label_shape'][1], 12), dtype=np.float32)
        else:
            label_map = np.zeros((self._geometry['label_shape'][0], self._geometry['label_shape'][1], 7), dtype=np.float32)
        label_list = []

        if self._distribution_setting == True:
            fp_bboxes = self._fp_distribution['fp_mean'][idx]   # 将均值作为期望的被检测框参数
            fp_var = self._fp_distribution['fp_var'][idx]
            fp_num = self._fp_distribution['fp_num'][idx]       # 用于筛选，总共392次实验，认为这一组中FP数量少于5的不作为FP
            scores = self._fp_distribution['fp_score'][idx]

            for i in range(len(fp_bboxes)):     # 处理一个FP框
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
        else:
            assert len(self._result) == len(self._gt_dt_matching_res['bev'])
            dt_bboxes = self._result[idx]['boxes_lidar']   # id帧下的所有检测框
            scores = self._result[idx]['score']    # id帧下的所有检测框得分
            matching_index = self._gt_dt_matching_res['bev'][idx]

            for i in range(len(dt_bboxes)):     # 处理一个检测框
                if i not in matching_index:     # 判断是否为FP
                    if scores[i] >= 0.3:
                        x = dt_bboxes[i][0]
                        y = dt_bboxes[i][1]
                        l = dt_bboxes[i][3]
                        w = dt_bboxes[i][4]
                        yaw = dt_bboxes[i][6]

                        corners, reg_target = self.get_corners([x, y, l, w, yaw], use_distribution=False)
                        self.update_label_map(label_map, corners, reg_target)
                        label_list.append(corners)

        return label_map, label_list

    def reg_target_transform(self, label_map):
        '''
        Inputs are numpy arrays (not tensors!)
        :param label_map: [352 * 400 * 7] label tensor
        :return: normalized regression map for all non_zero classification locations
        '''
        cls_map = label_map[..., 0]
        reg_map = label_map[..., 1:]

        index = np.nonzero(cls_map)
        reg_map[index] = (reg_map[index] - self.target_mean)/self.target_std_dev

    def __getitem__(self, index):
        assert index <= self.__len__()
        heatmap = self.get_heatmap(index)

        label_map, _ = self.get_label(index)
        # self.reg_target_transform(label_map)

        # occupancy = self.get_occupancy(index)
        # occlusion = self.get_occlusion(index)   # 直接使用pkl文件获取occ，方便调试；实际的推理过程中，没有预先处理好的occ，需要对GTbbox进行预处理获取
        occupancy, occlusion = self.get_occupancy_and_occlusion(index)       # 实际的推理过程中，使用该方法

        data_dict = {
            'occupancy': occupancy,
            'occlusion': occlusion,
            'heatmap': heatmap,
            'label_map': label_map
        }

        return data_dict

    def __len__(self):
        return len(self._result)

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
                if key in ['occupancy', 'occlusion', 'heatmap', 'label_map']:
                    heatmaps = []
                    for heatmap in val:
                        heatmaps.append(heatmap)
                    ret[key] = np.stack(heatmaps, axis=0)
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret
