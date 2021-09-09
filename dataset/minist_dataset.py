from dataset.dataset_base import DatasetBase
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MinistDataset(DatasetBase):
    def __init__(self, config):
        super().__init__()
        self._is_train = config['paras']['for_train']
        self._data_root = config['paras']['data_root']
        self._batch_size = config['paras']['batch_size']
        self._shuffle = config['paras']['shuffle']
        self._num_workers = config['paras']['num_workers']
        self._embedding_dataset = datasets.MNIST(  # train_set
            root=self._data_root,
            train=self._is_train,
            transform=transforms.ToTensor(),
            download=False
        )

    def __len__(self):
        return len(self._embedding_dataset)

    def __getitem__(self, item):
        assert item <= self.__len__()
        return self._embedding_dataset[item]

    def get_data_loader(self, distributed=False):
        return DataLoader(
            dataset=self,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
        )
