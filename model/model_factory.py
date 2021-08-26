# todo : change these import to a more automatic fashion
from model.model_base import ModelBase
from model.basic_module.cnn import CNN
from model.basic_module.mlp import MLP
from model.basic_module.vae import VAE
from model.top.vae_gan_model import VAEGANModel
from model.basic_module.mlp import MLP
from model.basic_module.vae import VAE
from model.basic_module.cnn import CNN
from utils.config.Configuration import Configuration

class ModelFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_config):
        class_name, paras = Configuration.find_dict_node(model_config, 'model_class')
        all_classes = ModelBase.__subclasses__()
        for cls in all_classes:
            if cls.__name__ == class_name:
                return cls(model_config['config_file']['expanded']) # todo not perfect
        raise TypeError(f'no class named \'{class_name}\' found in model folder')


