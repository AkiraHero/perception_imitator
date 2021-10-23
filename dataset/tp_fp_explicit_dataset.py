import pickle
import torch
import os
from dataset.dataset_base import DatasetBase
import numpy as np
from torch.utils.data import DataLoader


'''
Dataset for bounding boxes loading
    tp & fp explicit feature
    7_bbox_para, class, refl_u, refl_sigma, point_num
    label: tp ot fp (0 or 1)

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

        db_file = os.path.join(self._data_root, "tpfp_explicit_data.pkl")
        with open(db_file, 'rb') as f:
            self._tpfp = pickle.load(f)

        # train_data_size = int(len(self._tpfp) * 0.8)
        # if self._is_train == True:
        #     self.item_list = self._tpfp[:train_data_size]
        # else:
        #     self.item_list = self._tpfp[train_data_size:]

        # 为了得到所有图片中fp的质量，还是对于所有数据进行训练并测试
        self.item_list = self._tpfp

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
            if k in ['data']:
                v = torch.stack(v, 0)
                v = v.transpose(0,1).to(torch.float32)
                v = v.cuda()
                data[k] = v
            elif k in ['label']:
                v = v[0]
                v = v.cuda()
                data[k] = v