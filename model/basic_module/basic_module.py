import torch.nn as nn


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()

    @staticmethod
    def check_config(config):
        raise NotImplementedError

    @staticmethod
    def check_config_dict(required, config):
        assert isinstance(config, dict)
        for i in required:
            if i not in config.keys():
                raise KeyError

    @staticmethod
    def build_module(config):
        raise NotImplementedError
