from model.model_base import ModelBase
import factory.model_factory as ModelFactory
import torch

class VAEGANModel(ModelBase):
    def __init__(self, config):
        super(VAEGANModel, self).__init__()
        self.generator = ModelFactory.ModelFactory.get_model(config['paras']['submodules']['generator'])
        self.discriminator = ModelFactory.ModelFactory.get_model(config['paras']['submodules']['discriminator'])
        self.target_model = ModelFactory.ModelFactory.get_model(config['paras']['submodules']['target_model'])
        paras = torch.load(config['paras']['submodules']['target_model']['model_para_file'])
        self.target_model.load_state_dict(paras)

    def forward(self):
        raise NotImplementedError
