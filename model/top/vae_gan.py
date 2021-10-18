from model.model_base import ModelBase
import factory.model_factory as ModelFactory
import torch

class VAEGAN(ModelBase):
    def __init__(self, config):
        super(VAEGAN, self).__init__()
        self.generator = ModelFactory.ModelFactory.get_model(config['paras']['submodules']['generator'])
        self.discriminator = ModelFactory.ModelFactory.get_model(config['paras']['submodules']['discriminator'])

    def forward(self):
        raise NotImplementedError