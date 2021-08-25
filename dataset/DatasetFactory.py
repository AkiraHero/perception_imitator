from dataset.dataset_base import DatasetBase
from dataset.minist_dataset import MinistDataset


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_data_loader(data_config):
        class_name = data_config['dataset_class']
        all_classes = DatasetBase.__subclasses__()
        for cls in all_classes:
            if cls.__name__ == class_name:
                return cls(data_config['config_file']['expanded'])
        raise TypeError(f'no class named \'{class_name}\' found in dataset folder')
