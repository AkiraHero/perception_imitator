# add pvrcnn pretrained model results file and its gt matching file
import logging
import os
import pickle

import numpy as np
import torch

from model.model_base import ModelBase
from multiprocessing import Process, Manager


class FakePVRCNNOnKitti(ModelBase):
    class_map = {
        'Car': 1,
        'Pedestrian': 2,
        'Cyclist': 3
    }
    def __init__(self, config):
        super(FakePVRCNNOnKitti, self).__init__()
        self.kitti_results_folder = config['paras']['kitti_results_folder']
        self.num_workers = config['paras']['num_workers']
        self.max_obj_frame = 25
        if not os.path.isdir(self.kitti_results_folder):
            raise IsADirectoryError(f'The given folder({self.kitti_results_folder}) does not exist!')
        # check all pkl exists
        all_pkl = os.listdir(self.kitti_results_folder)
        if 7481 != len(all_pkl):
            raise FileExistsError("folder should contain 7481 pkl files.")
        logging.info("Using FakePVRCNNOnKitti: initialization success.")

    def _get_result(self, frm_id, res_dict):
        results_file = os.path.join(self.kitti_results_folder, f'{frm_id}.pkl')
        res = None
        with open(results_file, 'rb') as f:
            res = pickle.load(f)
            res_dict[frm_id] = res

    def _get_results(self, frm_id_unit, res_dict):
        for i in frm_id_unit:
            self._get_result(i, res_dict)

    def _split_ids(self, ids):
        work_size = len(ids)
        if work_size < self.num_workers:
            self.num_workers = work_size
        work_unit_size = work_size // self.num_workers
        work_unit_list = []
        idx = 0
        for i in range(self.num_workers):
            end_point = idx + work_unit_size if i < self.num_workers - 1 else len(ids)
            work_unit = ids[idx: end_point]
            work_unit_list.append(work_unit)
            idx = idx + work_unit_size
        return work_unit_list

    def _get_batch_result(self, data_batch):
        manager = Manager()
        result_dict = manager.dict()
        ids = data_batch['frame_id']
        id_units = self._split_ids(ids)
        proc_list = []
        for unit in id_units:
            p = Process(target=self._get_results, args=(unit, result_dict))
            proc_list.append(p)
            p.start()
        for p in proc_list:
            p.join()
        return result_dict

    def _extend2max_capacity(self, t):
        n = torch.zeros([self.max_obj_frame, 8], device=self.device)
        n[:t.shape[0], :] = t[:t.shape[0], :]
        return n

    def _data2device(self, data):
        # kitti train maximum obj in same frame is 24, so that we use a container with capacity of 25
        out_dict = {
            'gt_valid_inx': None,
            'dt_lidar_box': None,
            'frame_id': None
        }
        box_lidar = []
        gt_inx = []
        frame_id = []
        for k, frm in data.items():
            box_lidar.append(self._extend2max_capacity(torch.from_numpy(frm['ordered_lidar_boxes'])).unsqueeze(0))
            gt_inx.append(np.array(frm['gt_valid_inx']))
            frame_id.append(k)
        box_lidar = torch.cat(box_lidar, dim=0).to(self.device)
        gt_inx = np.array(gt_inx)
        out_dict['gt_valid_inx'] = gt_inx
        out_dict['dt_lidar_box'] = box_lidar
        out_dict['frame_id'] = frame_id
        return out_dict

    def forward(self, data_dict):
        data = dict(self._get_batch_result(data_dict))
        return self._data2device(data)
