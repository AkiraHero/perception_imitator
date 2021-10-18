import pickle
import torch
import os
from dataset.dataset_base import DatasetBase
import numpy as np
from torch.utils.data import DataLoader


'''
Dataset for bounding boxes loading
    tp & fp explicit feature

'''

class TpFpDataset(DatasetBase):
    def __init__(self, config):
        super(DatasetBase, self).__init__()
        load_all = False if "load_all" not in config['paras'].keys() else config['paras']['load_all']
        screen_no_dt = False if "screen_no_dt" not in config['paras'].keys() else config['paras']['screen_no_dt']
        self._is_train = config['paras']['for_train']
        self._batch_size = config['paras']['batch_size']
        self._data_root = config['paras']['data_root']
        self._num_workers = config['paras']['num_workers']
        self._shuffle = config['paras']['shuffle']

        db_file = os.path.join(self._data_root, "gt_dt_matching_res.pkl")
        with open(db_file, 'rb') as f:
            self._db = pickle.load(f)

        # self.item_list = torch.Tensor(self._fpbox['fp_bbox'])

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
