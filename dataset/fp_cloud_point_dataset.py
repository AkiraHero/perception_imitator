import pickle
import torch
import os
from dataset.dataset_base import DatasetBase
import numpy as np
from torch.utils.data import DataLoader


'''
Dataset including:
    FP's bbox parameter
    point cloud in FP

'''

class FpCloudPointDataset(DatasetBase):
    def __init__(self, config):
        super(DatasetBase, self).__init__()
        load_all = False if "load_all" not in config['paras'].keys() else config['paras']['load_all']
        screen_no_dt = False if "screen_no_dt" not in config['paras'].keys() else config['paras']['screen_no_dt']
        self._is_train = config['paras']['for_train']
        self._batch_size = config['paras']['batch_size']
        self._data_root = config['paras']['data_root']
        self._num_workers = config['paras']['num_workers']
        self._shuffle = config['paras']['shuffle']
        fp_cloud_point_file = os.path.join(self._data_root, "fp_cloud_point.pkl")
        with open(fp_cloud_point_file, 'rb') as f:
            self._fp_cp = pickle.load(f)

        train_data_size = int(len(self._fp_cp) * 0.8)
        if self._is_train == True:
            self.item_list = self._fp_cp[:train_data_size]
        else:
            self.item_list = self._fp_cp[train_data_size:]

    def get_data_loader(self, distributed=False):
        return DataLoader(
            dataset=self,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
        )

    def __getitem__(self, index):
        assert index <= self.__len__()
        return self.item_list[index]

    def __len__(self):
        return len(self.item_list)
