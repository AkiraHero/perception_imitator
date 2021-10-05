import pickle
import torch
import os
from dataset.dataset_base import DatasetBase
import numpy as np
from torch.utils.data import DataLoader


'''
Dataset for bounding boxes loading
    ground truth boxes of kitti dataset
    detection result of PVRCNN pretrained model

'''


class BoxOnlyDataset(DatasetBase):
    def __init__(self, config):
        super(DatasetBase, self).__init__()
        load_all = None if "load_all" not in config['paras'].keys() else config['paras']['load_all']
        screen_no_dt = False if "screen_no_dt" not in config['paras'].keys() else config['paras']['screen_no_dt']
        self._is_train = config['paras']['for_train']
        self._batch_size = config['paras']['batch_size']
        self._data_root = config['paras']['data_root']
        self._data_workers = config['paras']['num_workers']
        db_file = os.path.join(self._data_root, "kitti_pvrcnn_all.pkl")
        with open(db_file, 'rb') as f:
            self._db = pickle.load(f)
        self._train_frm_ratio = 0.6
        train_test_dataset_division_file = os.path.join(self._data_root, "train_test_sample.pkl")
        train_test_dataset_division = None
        if os.path.exists(train_test_dataset_division_file):
            with open(train_test_dataset_division_file, 'rb') as f:
                train_test_dataset_division = pickle.load(f)
                train_list = train_test_dataset_division['train_list']
                test_list = train_test_dataset_division['test_list']

        else:
            train_list, test_list = self.sample_dataset()
            d = {
                'train_list': train_list,
                'test_list': test_list
            }
            with open(train_test_dataset_division_file, 'wb') as f:
                pickle.dump(d, f)
        self.train_db = [self._db[i] for i in train_list]
        self.test_db = [self._db[i] for i in test_list]
        if load_all is None or load_all is False:
            if self._is_train:
                self.item_list = BoxOnlyDataset.get_data_dict(self.train_db)
                pass
            else:
                self.item_list = BoxOnlyDataset.get_data_dict(self.test_db)
        else:
            self.item_list = BoxOnlyDataset.get_data_dict(self._db)
        if screen_no_dt:
            self.item_list = [i for i in self.item_list if i['detected']]
            pass

        # transform continuous regression to discrete classfication
        get_discrete_cls = 1
        discrete_cls_num = 6
        if get_discrete_cls:
            # get segment
            all_item_list = BoxOnlyDataset.get_data_dict(self._db)
            box_diff = np.concatenate([i['box_diff'].reshape(1, -1) for i in all_item_list if i['detected']], axis=0)
            segs = self.get_error_segs(box_diff, class_num=discrete_cls_num)
            self.segs = segs
            for i in self.item_list:
                i['box_diff_cls'] = np.zeros([7, 1])
                for j in range(7):
                    cls = self.get_discrete_type(i['box_diff'][j], segs[j])
                    i['box_diff_cls'][j, 0] = cls
        pass

    def __getitem__(self, index):
        return self.item_list[index]

    def __len__(self):
        return len(self.item_list)

    @staticmethod
    def get_discrete_type(value, seglist):
        if value <= seglist[0][0]:
            return 0
        if value >= seglist[-1][1]:
            return len(seglist) - 1
        for inx, i in enumerate(seglist):
            if i[0] <= value < i[1]:
                return inx
        raise NotImplementedError

    @staticmethod
    def get_error_segs(box_diff, class_num=4):
        seg_list = []
        for i in range(7):
            sub_seg_list = []
            sorted_vec = np.sort(box_diff[:, i])
            length = len(sorted_vec)
            step = length // class_num
            now = 0
            for j in range(class_num):
                ed = now + step
                if ed >= length:
                    ed = length - 1
                sub_seg_list.append([sorted_vec[now], sorted_vec[ed]])
                now = now + step
            seg_list.append(sub_seg_list)
        return seg_list

    @staticmethod
    def get_discrete_distribution_diff(sq_target, sq_model, seg_list):
        seg_target_list = []
        seg_predict_list = []
        seg_intersection_list = []
        for i in seg_list:
            model_inx, = ((sq_model >= i[0]) & (sq_model < i[1])).nonzero()
            target_inx, = ((sq_target >= i[0]) & (sq_target < i[1])).nonzero()
            seg_target_list.append(len(target_inx))
            seg_predict_list.append(len(model_inx))
            seg_intersection_list.append(len(np.intersect1d(target_inx, model_inx)))
        return seg_target_list, seg_predict_list, seg_intersection_list

    def sample_dataset(self):
        frm_num = len(self._db)
        train_list = []
        test_list = []
        for i in range(frm_num):
            if np.random.rand() < self._train_frm_ratio:
                train_list.append(i)
            else:
                test_list.append(i)
        return train_list, test_list

    @staticmethod
    def batch_collate_fn(data):
        batch_size = len(data)
        keys = data[0].keys()
        batch_data_dict = {
            key: np.array([i[key] for i in data]).reshape(batch_size, -1) for key in keys if key != 'box_diff_cls'
        }
        batch_data_dict['box_diff_cls'] = np.array([i['box_diff_cls'] for i in data])
        return batch_data_dict

    def get_data_loader(self):
        data_loader = DataLoader(
            dataset=self,
            batch_size=self._batch_size,
            shuffle=True if self._is_train else False,
            num_workers=self._data_workers,
            collate_fn=self.batch_collate_fn
        )
        return data_loader

    @staticmethod
    def load_data2gpu(data):
        for k, v in data.items():
            v = torch.from_numpy(v).to(torch.float)
            v = v.cuda()
            data[k] = v

    @staticmethod
    def label_str2num(label):
        d = {
            'Car': 1,
            'Pedestrian': 2,
            'Cyclist': 3
        }
        return d[label]

    @staticmethod
    def get_data_dict(db):
        # sum up all frm
        datadict_list = []
        for frm in db:
            for inx, matched_obj_inx in enumerate(frm['gt_valid_inx']):
                gt = frm['ini_frm_annos']['gt_boxes_lidar']
                cur_gt_box = gt[matched_obj_inx]
                cur_dt_box = frm['ordered_lidar_boxes'][inx]
                cur_gt_label = BoxOnlyDataset.label_str2num(frm['ini_frm_annos']['name'][matched_obj_inx])
                if np.sum(cur_dt_box).astype(np.int) == 0:
                    detected = 0
                else:
                    detected = 1
                box_diff = cur_dt_box[:7] - cur_gt_box
                rot_diff = np.arctan(np.tan(box_diff[6]))
                box_diff[6] = rot_diff
                cur_gt_box[6] = np.arctan(np.tan(cur_gt_box[6]))
                cur_dt_box[6] = np.arctan(np.tan(cur_dt_box[6]))
                data_dict = {'detected': detected, 'box_diff': box_diff,
                             'gt_box': np.concatenate([cur_gt_box, np.array([cur_gt_label])]), 'dt_box': cur_dt_box}
                datadict_list.append(data_dict)
        return datadict_list