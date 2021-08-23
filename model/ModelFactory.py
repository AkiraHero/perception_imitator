# todo : change these import to a more automatic fashion
from model.model_base import ModelBase


class ModelFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_config):
        class_name = model_config['model_class']
        all_classes = ModelBase.__subclasses__()
        for cls in all_classes:
            if cls.__name__ == class_name:
                return cls(model_config)
        raise TypeError(f'no class named \'{class_name}\' found in dataset folder')


