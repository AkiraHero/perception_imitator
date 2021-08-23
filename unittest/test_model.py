from model.basic_module.mlp import MLP
from model.basic_module.cnn import CNN
from model.model_base import ModelBase

import yaml

ModelBase.register_class()


config_file = "/home/xlju/Project/ModelSimulator/utils/config/model/samples/mlp.yaml"
with open(config_file, 'r') as f:
    config = yaml.load(f)
    sample_mlp = MLP.build_module(config)
    pass

config_file = "/home/xlju/Project/ModelSimulator/utils/config/model/samples/cnn.yaml"
with open(config_file, 'r') as f:
    config = yaml.load(f)
    sample_cnn = CNN.build_module(config)
    pass