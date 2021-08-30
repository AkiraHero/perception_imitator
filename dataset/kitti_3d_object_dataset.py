from dataset.dataset_base import DatasetBase
from torch.utils.data import DataLoader
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from easydict import EasyDict as edict
from pathlib import Path

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

    def get_data_loader(self):
        return DataLoader(
            dataset=self,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
        )

    def get_embedding(self):
        return self._embedding_dataset
