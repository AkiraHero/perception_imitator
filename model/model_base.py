import torch.nn as nn


class ModelBase(nn.Module):
    module_class_list = {}

    def __init__(self):
        super(ModelBase, self).__init__()
        self.output_shape = None
        self.device = None
        self.mod_dict = nn.ModuleDict()

    def get_output_shape(self):
        if not self.output_shape:
            raise NotImplementedError
        return self.output_shape

    def set_device(self, device):
        self.device = device
        for k, v in self.mod_dict.items():
            if 'set_device' in v.__dir__():
                v.set_device(device)
        self.to(self.device)

    def set_eval(self):
        self.eval()
        for k, v in self.mod_dict.items():
            if 'eval' in v.__dir__():
                v.eval()

    @staticmethod
    def check_config(config):
        required_paras = ['name', 'paras']
        ModelBase.check_config_dict(required_paras, config)

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
