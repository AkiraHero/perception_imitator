import pickle
import torch
import os
import math
from dataset.dataset_base import DatasetBase
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict

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
        
        occ_file = os.path.join(self._data_root, "easy_scene_temp.pkl")
        with open(occ_file, 'rb') as f:
            self._occ = pickle.load(f)
        heatmap_file = os.path.join(self._data_root, "GTheatmap_aug.pkl")
        with open(heatmap_file, 'rb') as f:
            self._heatmap = pickle.load(f)[7000:7481]
        print("Dataset不全，目前只有7000-7481")
        
        exp_name = config['paras']['exp_name_root']
        matching_name = os.path.join(exp_name, 'gt_dt_matching_res.pkl')
        result_name = os.path.join(exp_name, 'result.pkl')
        with open(matching_name, 'rb') as f:
            self._gt_dt_matching_res = pickle.load(f)
        with open(result_name, 'rb') as f:
            self._result = pickle.load(f)

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

    def get_corners(self, bbox):
        x, y, l, w, yaw = bbox        
        #x, y, w, l, yaw = self.interpret_kitti_label(bbox)
        
        bev_corners = np.zeros((4, 2), dtype=np.float32)
        # front left
        bev_corners[0, 0] = x - l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[0, 1] = y - l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        # rear left
        bev_corners[1, 0] = x - l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[1, 1] = y - l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # rear right
        bev_corners[2, 0] = x + l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[2, 1] = y + l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front right
        bev_corners[3, 0] = x + l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[3, 1] = y + l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        reg_target = [np.cos(yaw), np.sin(yaw), x, y, w, l]

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
            actual_reg_target[4] = np.log(reg_target[4])
            actual_reg_target[5] = np.log(reg_target[5])

            map[label_x, label_y, 0] = 1.0
            map[label_x, label_y, 1:7] = actual_reg_target

    def get_occupancy(self, idx):
        return self._occ[idx]['occupancy']

    def get_occlusion(self, idx):
        return self._occ[idx]['occlusion']

    def get_heatmap(self, idx):
        return self._heatmap[idx]['gt_heatmap'] / 255

    def get_label(self, idx):
        label_map = np.zeros(self._geometry['label_shape'], dtype=np.float32)
        label_list = []

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

                    corners, reg_target = self.get_corners([x, y, l, w, yaw])
                    self.update_label_map(label_map, corners, reg_target)
                    label_list.append(corners)

        return label_map, label_list

    def __getitem__(self, index):
        assert index <= self.__len__()

        occupancy = self.get_occupancy(index)
        occlusion = self.get_occlusion(index)
        heatmap = self.get_heatmap(index)
        label_map, _ = self.get_label(index)

        data_dict = {
            'occupancy': occupancy,
            'occlusion': occlusion,
            'heatmap': heatmap,
            'label_map': label_map
        }

        return data_dict

    def __len__(self):
        assert len(self._occ) == len(self._heatmap)
        # assert len(self._occ) == len(self._result)
        return len(self._occ)

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
