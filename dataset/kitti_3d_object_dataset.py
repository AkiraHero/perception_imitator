from dataset.dataset_base import DatasetBase
from torch.utils.data import DataLoader
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from easydict import EasyDict as edict
from pathlib import Path
import torch
import numpy as np
from collections import defaultdict
import kornia


'''
    The dataset gives data including following keys:
        'frame_id', 
        'calib', 
        'gt_boxes', 
        'points', 
        'use_lead_xyz',
        'voxels', 
        'voxel_coords', 
        'voxel_num_points', 
        'image_shape', 
        'batch_size'
'''

class Kitti3dObjectDataset(DatasetBase):
    def __init__(self, config):
        super().__init__()
        self._is_train = config['paras']['for_train']
        self._data_root = config['paras']['data_root']
        self._batch_size = config['paras']['batch_size']
        self._shuffle = config['paras']['shuffle']
        self._num_workers = config['paras']['num_workers']
        self._embedding_dataset = KittiDataset(
            dataset_cfg=edict(config['paras']['config_file']['expanded']),
            class_names=config['paras']['class_names'],
            root_path=Path(self._data_root),
            training=config['paras']['for_train'],
            logger=None,
        )

    def __len__(self):
        return len(self._embedding_dataset)

    def __getitem__(self, item):
        assert item <= self.__len__()
        return self._embedding_dataset[item]

    def get_data_loader(self, distributed=False):
        if distributed:
            if self._is_train:
                sampler = torch.utils.data.distributed.DistributedSampler(self)
            else:
                raise NotImplementedError
        else:
            sampler = None

        return DataLoader(
            dataset=self,
            batch_size=self._batch_size,
            shuffle=(sampler is None) and self._shuffle,
            num_workers=self._num_workers,
            pin_memory=True,
            collate_fn=Kitti3dObjectDataset.collate_batch,
            drop_last=False,
            sampler=sampler,
            timeout=0
        )

    def get_embedding(self):
        return self._embedding_dataset

    @staticmethod
    def load_data_to_gpu(batch_dict):
        for key, val in batch_dict.items():
            if not isinstance(val, np.ndarray):
                continue
            elif key in ['frame_id', 'metadata', 'calib']:
                continue
            elif key in ['images']:
                batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
            elif key in ['image_shape']:
                batch_dict[key] = torch.from_numpy(val).int().cuda()
            else:
                batch_dict[key] = torch.from_numpy(val).float().cuda()

    @staticmethod
    def default_collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ["images", "depth_maps"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = Kitti3dObjectDataset.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = Kitti3dObjectDataset.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        # Pad with nan, to be replaced later in the pipeline.
                        pad_value = np.nan

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(image,
                                           pad_width=pad_width,
                                           mode='constant',
                                           constant_values=pad_value)

                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0)
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret


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
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ["images", "depth_maps"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = Kitti3dObjectDataset.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = Kitti3dObjectDataset.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        # Pad with nan, to be replaced later in the pipeline.
                        pad_value = np.nan

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(image,
                                           pad_width=pad_width,
                                           mode='constant',
                                           constant_values=pad_value)

                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0)
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret

    @staticmethod
    def get_pad_params(desired_size, cur_size):
        """
        Get padding parameters for np.pad function
        Args:
            desired_size: int, Desired padded output size
            cur_size: int, Current size. Should always be less than or equal to cur_size
        Returns:
            pad_params: tuple(int), Number of values padded to the edges (before, after)
        """
        assert desired_size >= cur_size

        # Calculate amount to pad
        diff = desired_size - cur_size
        pad_params = (0, diff)

        return pad_params
