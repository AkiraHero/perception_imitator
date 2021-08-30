from model.model_base import ModelBase
from dataset.dataset_base import DatasetBase
from factory.dataset_factory import DatasetFactory
from pcdet.models.detectors.pv_rcnn import PVRCNN
import torch.nn as nn
from easydict import EasyDict


class OpenPCDetPVRCNN(ModelBase):
    def __init__(self, config):
        super(OpenPCDetPVRCNN, self).__init__()
        self.dataset_ref = None
        self._config = config
        self.embedding_model = None
        self.set_attr("dataset", DatasetFactory.get_singleton_dataset())
        self._set_embedding_model()
        if not isinstance(self.embedding_model, nn.Module):
            raise TypeError('Embedding model has to be a subclass of nn.Module to ensure torch hooks work normally.')

    def set_attr(self, attr, value):
        if attr == "dataset":
            if not isinstance(value, DatasetBase):
                raise TypeError("The instance set for dataset has been checked as non-dataset class.")
            self.dataset_ref = value

    def _set_embedding_model(self):
        if self.dataset_ref is not None:
            self.embedding_model = PVRCNN(
                model_cfg=EasyDict(self._config['paras']['config_file']['expanded']['MODEL']),
                num_class=len(self._config['paras']['config_file']['expanded']['CLASS_NAMES']),
                dataset=self.dataset_ref.get_embedding()
            )

    def forward(self, batch_dict):
        if self.embedding_model is None:
            raise ModuleNotFoundError("Please call set_attr to set a dataset reference for this model.")
        return self.embedding_model.forward(batch_dict)

    def load_model_paras(self, para_dict):
        return self.embedding_model.load_state_dict(para_dict['model_state'])
