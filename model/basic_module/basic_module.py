import torch.nn as nn


class BasicModule(nn.Module):
    module_class_list = {}

    def __init__(self):
        super(BasicModule, self).__init__()
        self.output_shape = None

    def get_output_shape(self):
        if not self.output_shape:
            raise NotImplementedError
        return self.output_shape

    @staticmethod
    def check_config(config):
        required_paras = ['name', 'paras']
        BasicModule.check_config_dict(required_paras, config)

    @staticmethod
    def check_config_dict(required, config):
        assert isinstance(config, dict)
        for i in required:
            if i not in config.keys():
                err = f'Required config {i} does not exist.'
                raise KeyError(err)

    @staticmethod
    def build_module(config):
        raise NotImplementedError

    @classmethod
    def register_class(cls):
        for i in cls.__subclasses__():
            cls.module_class_list[i.__name__] = i
            print(i.__name__)
