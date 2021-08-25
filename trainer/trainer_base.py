from model.model_base import ModelBase
from dataset.dataset_base import DatasetBase


class TrainerBase:
    def __init__(self):
        self.model = None
        self.dataset = None
        self.optimizer = None
        pass

    def check_ready(self):
        if self.model is None:
            return False
        if self.dataset is None:
            return False
        return True

    def run(self):
        raise NotImplementedError

    def set_model(self, model):
        if not isinstance(model, ModelBase):
            raise TypeError
        self.model = model

    def set_dataset(self, dataset):
        if not isinstance(dataset, DatasetBase):
            raise TypeError
        self.dataset = dataset

    def set_optimizer(self, optimizer_config):
        raise NotImplementedError


