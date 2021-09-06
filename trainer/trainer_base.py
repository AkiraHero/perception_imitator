from model.model_base import ModelBase
from dataset.dataset_base import DatasetBase
from utils.logger.basic_logger import BasicLogger


class TrainerBase:
    def __init__(self):
        self.model = None
        self.dataset = None
        self.optimizer = None
        self.max_epoch = 0
        self.epoch = 0
        self.step = 0
        self.global_step = 0
        self.logger = None

    def get_training_status(self):
        training_status = {
            "max_epoch": self.max_epoch,
            "epoch": self.epoch,
            "step": self.step,
            "global_step": self.global_step
        }
        return training_status

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

    def load_state(self, state_dict):
        raise NotImplementedError

    def save_state(self, state_dict):
        raise NotImplementedError

    def set_logger(self, logger):
        if not isinstance(logger, BasicLogger):
            raise TypeError("logger must be with the type: BasicLogger")
        self.logger = logger
        self.logger.register_status_hook(self.get_training_status)


