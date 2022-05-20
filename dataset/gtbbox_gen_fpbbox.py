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
        
        allgt2fp_file = os.path.join(self._data_root, "carla_gtbbox_gen_20fpbbox.pkl")
        with open(allgt2fp_file, 'rb') as f:
            self._allgt2fp = pickle.load(f)

        self.gt_mean, self.gt_std, self.fp_mean, self.fp_std = self.get_standardize_params(self._allgt2fp)

        train_data_size = int(len(self._allgt2fp) * 0.8)
        if self._is_train == True:
            self.item_list = self._allgt2fp[:train_data_size]
        else:
            self.item_list = self._allgt2fp[train_data_size:]

        self.standardize()


    def standardize(self):
        for index in range(len(self.item_list)):
            gt_bboxes = np.array(self.item_list[index]['gt_bboxes']).reshape(-1, 8)
            fp_bboxes = np.array(self.item_list[index]['fp_bboxes_all']).reshape(-1, 7)
            gt_mask = np.sum(gt_bboxes, axis=1)!=0
            fp_mask = np.sum(fp_bboxes, axis=1)!=0

            gt_bboxes[gt_mask][:, 0:7] = np.where(np.isnan(gt_bboxes[gt_mask][:, 0:7]), np.array(np.nan), (gt_bboxes[gt_mask][:, 0:7] - self.gt_mean[0:7]) / self.gt_std[0:7])
            fp_bboxes[fp_mask] = np.where(np.isnan(fp_bboxes[fp_mask]), np.array(np.nan), (fp_bboxes[fp_mask] - self.fp_mean) / self.fp_std)

            self.item_list[index].update(gt_bboxes = gt_bboxes.reshape(640).astype('float32').tolist())
            self.item_list[index].update(fp_bboxes_all = fp_bboxes.reshape(140).astype('float32').tolist())

    def unstandardize(self, array, mean=None, std=None):
        return array * std + mean

    def get_standardize_params(self, list):
        all_gt_bboxes = []
        all_fp_bboxes = []
        for dict in list:
            all_gt_bboxes.append(dict['gt_bboxes'])
            all_fp_bboxes.append(dict['fp_bboxes_all'])

        all_gt_bboxes = np.stack(all_gt_bboxes, axis=0).reshape(-1, 8)
        all_fp_bboxes = np.stack(all_fp_bboxes, axis=0).reshape(-1, 7)
        
        gt_mask = np.sum(all_gt_bboxes, axis=1)!=0
        fp_mask = np.sum(all_fp_bboxes, axis=1)!=0

        gt_mean = np.mean(all_gt_bboxes[gt_mask], axis = 0)
        gt_std = np.std(all_gt_bboxes[gt_mask], axis = 0)
        fp_mean = np.mean(all_fp_bboxes[fp_mask], axis = 0)
        fp_std = np.std(all_fp_bboxes[fp_mask], axis = 0)
        
        return gt_mean, gt_std, fp_mean, fp_std   

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
            if k in ['gt_bboxes', 'tp_bboxes', 'fp_bboxes_all', 'fp_bboxes_easy', 'fp_bboxes_hard', 'difficult']:
                v = torch.stack(v, 0)
                v = v.transpose(0,1).to(torch.float32)
                v = v.cuda()
                data[k] = v
            else: 
                pass
