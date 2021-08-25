from model.model_base import ModelBase
import model.ModelFactory as ModelFactory
import torch

class VAEGANModel(ModelBase):
    def __init__(self, config):
        super(VAEGANModel, self).__init__()
        self.encoder = ModelFactory.ModelFactory.get_model(config['paras']['submodules']['encoder'])
        self.discriminator = ModelFactory.ModelFactory.get_model(config['paras']['submodules']['discriminator'])
        self.target_model = ModelFactory.ModelFactory.get_model(config['paras']['submodules']['target_model'])
        self.mod_dict.add_module('encoder', self.encoder)
        self.mod_dict.add_module('discriminator', self.discriminator)
        self.mod_dict.add_module('target_model', self.target_model)
        paras = torch.load(config['paras']['submodules']['target_model']['model_para_file'])
        self.target_model.load_state_dict(paras)

    def forward(self):
        raise NotImplementedError
