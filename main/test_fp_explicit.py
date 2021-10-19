import sys
sys.path.append('D:/1Pjlab/ModelSimulator/')
from utils.config.Configuration import Configuration
from factory.model_factory import ModelFactory
from factory.dataset_factory import DatasetFactory
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch
import numpy as np

right_prediction_dict_target = OrderedDict()
right_prediction_dict_gen = OrderedDict()
whole_class_num = OrderedDict()
precision_dict_target = OrderedDict()
precision_dict_gen = OrderedDict()

for i in range(10):
    right_prediction_dict_target[i] = 0
    right_prediction_dict_gen[i] = 0
    whole_class_num[i] = 0

if __name__ == '__main__':
    # manage config
    config = Configuration()
    args = config.get_shell_args_train()
    args.for_train = False
    args.shuffle = False
    config.load_config(args.cfg_dir)
    config.overwrite_config_by_shell_args(args)

    # instantiating all modules by non-singleton factory
    model = ModelFactory.get_model(config.model_config)

    # paras_final = torch.load("/home/xlju/Project/VAE_Mnist2/model_50.pt")
    # ref = model.generator.state_dict()
    # for (k1, v1), (k2, v2) in zip(ref.items(), paras_final.items()):
    #     ref[k1] = v2

    paras = torch.load("D:/1Pjlab/ModelSimulator/output/fp_explicit_model/310.pt")
    model.load_model_paras(paras)
    model.set_eval()
    dataset = DatasetFactory.get_dataset(config.dataset_config)
    data_loader = dataset.get_data_loader()

    accuracy_add = 0
    for step, data in enumerate(data_loader):
        explicit_data = data[:,:-1].float()
        label = data[:,-1].long()

        pred = model(explicit_data)
        _, indices = torch.max(pred, 1)

        accuracy = np.array((indices == label)).tolist().count(True)/ len(np.array(label).tolist())
        accuracy_add = accuracy_add + accuracy

        print("accuracy =", accuracy)
        print("step =", step, )

    all_accuracy = accuracy_add/(step + 1)
    print("all_accuracy =", all_accuracy)

    pass
