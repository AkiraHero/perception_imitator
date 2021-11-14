import pickle
import torch
import os
from dataset.dataset_base import DatasetBase
import numpy as np
from torch.utils.data import DataLoader


'''
Dataset for bounding boxes loading
    data: all gt bboxes representing the scene
    label: FP bboxes in detection

'''

class GtbboxGenFpbbox(DatasetBase):
    def __init__(self, config):
        super(DatasetBase, self).__init__()
        load_all = False if "load_all" not in config['paras'].keys() else config['paras']['load_all']
        screen_no_dt = False if "screen_no_dt" not in config['paras'].keys() else config['paras']['screen_no_dt']
        self._is_train = config['paras']['for_train']
        self._batch_size = config['paras']['batch_size']
        self._data_root = config['paras']['data_root']
        self._num_workers = config['paras']['num_workers']
        self._shuffle = config['paras']['shuffle']
        
        allgt2fp_file = os.path.join(self._data_root, "gtbbox_gen_fpbbox.pkl")
        with open(allgt2fp_file, 'rb') as f:
            self._allgt2fp = pickle.load(f)

        train_data_size = int(len(self._allgt2fp) * 0.8)
        if self._is_train == True:
            self.item_list = self._allgt2fp[:train_data_size]
        else:
            self.item_list = self._allgt2fp[train_data_size:]

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

    @staticmethod
    def load_data2gpu(data):
        for k, v in data.items():
            if k in ['gt_bboxes', 'fp_bboxes']:
                v = torch.stack(v, 0)
                v = v.transpose(0,1).to(torch.float32)
                v = v.cuda()
                data[k] = v
            else:
                v = v.cuda()
                data[k] = v
