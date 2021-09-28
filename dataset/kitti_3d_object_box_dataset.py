from dataset.dataset_base import DatasetBase
from torch.utils.data import DataLoader
import os
import pickle
import numpy as np

class Kitti3dObjectBoxDataset(DatasetBase):
    def __init__(self, config):
        super().__init__()
        self._is_train = config['paras']['for_train']
        self._data_root = config['paras']['data_root']
        self._batch_size = config['paras']['batch_size']
        self._shuffle = config['paras']['shuffle']
        self._num_workers = config['paras']['num_workers']
        self._data_pkl = os.path.join(self._data_root, "all_boxes_kitti.pkl")
        self._data_mem = None
        with open(self._data_pkl, 'rb') as f:
            self._data_mem = pickle.load(f)
        pass


    def __len__(self):
        return len(self._data_mem.keys())

    def __getitem__(self, item):
        assert item <= self.__len__()
        item_inx_key = "{:0>6d}".format(item)
        assert item_inx_key in self._data_mem.keys()
        return self._data_mem[item_inx_key]

    def get_data_loader(self, distributed=False):
        return DataLoader(
            dataset=self,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            collate_fn=self.default_collate_batch
        )

    @staticmethod
    def load_data_to_gpu(batch_dict):
        # for key, val in batch_dict.items():
        #     if not isinstance(val, np.ndarray):
        #         continue
        #     elif key in ['frame_id', 'metadata', 'calib', 'point_inx']:
        #         continue
        #     elif key in ['images']:
        #         batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        #     elif key in ['image_shape']:
        #         batch_dict[key] = torch.from_numpy(val).int().cuda()
        #     else:
        #         batch_dict[key] = torch.from_numpy(val).float().cuda()
        pass

    @staticmethod
    def default_collate_batch(batch_list, _unused=False):
        ret = {
            'box_num': [],
            'gt_box': [],
            'dt_box': [],
            'dt_valid': [],
            'frame_id': []
        }
        batch_size = len(batch_list)
        box_max = 25
        for i in batch_list:
            ret['box_num'].append(len(i['gt_valid_inx']))
            ret['frame_id'].append(i['frame_id'])
            ret['gt_box'].append(i['ordered_gt_boxes'])
            ret['dt_box'].append(i['ordered_lidar_boxes'])

        batch_gt_boxes3d = np.zeros((batch_size, box_max, ret['gt_box'][0].shape[-1]), dtype=np.float32)
        for i, k in enumerate(ret['gt_box']):
            batch_gt_boxes3d[i, :k.shape[0], :] = k

        batch_dt_boxes3d = np.zeros((batch_size, box_max, ret['dt_box'][0].shape[-1]), dtype=np.float32)
        for i, k in enumerate(ret['dt_box']):
            batch_dt_boxes3d[i, :k.shape[0], :] = k
        ret['gt_box'] = batch_gt_boxes3d
        ret['dt_box'] = batch_dt_boxes3d
        pass
        # for cur_sample in batch_list:
        #     for key, val in cur_sample.items():
        #         data_dict[key].append(val)
        # batch_size = len(batch_list)
        # ret = {}
        #
        # for key, val in data_dict.items():
        #     try:
        #         if key in ['voxels', 'voxel_num_points']:
        #             ret[key] = np.concatenate(val, axis=0)
        #         elif key in ['points', 'voxel_coords']:
        #             coors = []
        #             for i, coor in enumerate(val):
        #                 coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
        #                 coors.append(coor_pad)
        #             ret[key] = np.concatenate(coors, axis=0)
        #         elif key in ['gt_boxes']:
        #             max_gt = max([len(x) for x in val])
        #             batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
        #             for k in range(batch_size):
        #                 batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
        #             ret[key] = batch_gt_boxes3d
        #         elif key in ['gt_boxes2d']:
        #             max_boxes = 0
        #             max_boxes = max([len(x) for x in val])
        #             batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
        #             for k in range(batch_size):
        #                 if val[k].size > 0:
        #                     batch_boxes2d[k, :val[k].__len__(), :] = val[k]
        #             ret[key] = batch_boxes2d
        #         elif key in ["images", "depth_maps"]:
        #             # Get largest image size (H, W)
        #             max_h = 0
        #             max_w = 0
        #             for image in val:
        #                 max_h = max(max_h, image.shape[0])
        #                 max_w = max(max_w, image.shape[1])
        #
        #             # Change size of images
        #             images = []
        #             for image in val:
        #                 pad_h = Kitti3dObjectDataset.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
        #                 pad_w = Kitti3dObjectDataset.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
        #                 pad_width = (pad_h, pad_w)
        #                 # Pad with nan, to be replaced later in the pipeline.
        #                 pad_value = np.nan
        #
        #                 if key == "images":
        #                     pad_width = (pad_h, pad_w, (0, 0))
        #                 elif key == "depth_maps":
        #                     pad_width = (pad_h, pad_w)
        #
        #                 image_pad = np.pad(image,
        #                                    pad_width=pad_width,
        #                                    mode='constant',
        #                                    constant_values=pad_value)
        #
        #                 images.append(image_pad)
        #             ret[key] = np.stack(images, axis=0)
        #         else:
        #             ret[key] = np.stack(val, axis=0)
        #     except:
        #         print('Error in collate_batch: key=%s' % key)
        #         raise TypeError
        #
        # ret['batch_size'] = batch_size
        # return ret

