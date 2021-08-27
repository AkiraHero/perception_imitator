from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class DatasetBase(Dataset):
    def __init__(self):
        super(DatasetBase, self).__init__()

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_data_loader(self):
        raise NotImplementedError
