# todo : change these import to a more automatic fashion
from model.basic_module.basic_module import BasicModule
from model.basic_module.mlp import MLP
from model.basic_module.cnn import CNN
from model.top.vae_gan_model import VAEGANModel


class ModelFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_config):
        class_name = model_config['model_class']
        all_classes = BasicModule.__subclasses__()
        for cls in all_classes:
            if cls.__name__ == class_name:
                return cls(model_config)
        raise TypeError(f'no class named \'{class_name}\' found in dataset folder')


