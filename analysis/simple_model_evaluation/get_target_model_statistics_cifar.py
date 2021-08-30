from utils.config.Configuration import Configuration
from factory.model_factory import ModelFactory
from factory.dataset_factory import DatasetFactory
from collections import OrderedDict
import torch
import matplotlib.pyplot as plt


'''
reproduce the minist classifier precision curve
using config sample: /home/xlju/Project/ModelSimulator/utils/config/samples/sample2
Usage: python *.py --cfg_dir /home/xlju/Project/ModelSimulator/utils/config/samples/sample2
'''

right_prediction_dict = OrderedDict()
whole_class_num = OrderedDict()
precision_dict = OrderedDict()
for i in range(10):
    right_prediction_dict[i] = 0
    whole_class_num[i] = 0


def set_to_counter(model_out, gt):
    res = model_out == gt
    for i, j in zip(res, gt):
        k = j.item()
        if i:
            right_prediction_dict[k] = right_prediction_dict[k] + 1
        whole_class_num[k] = whole_class_num[k] + 1



if __name__ == '__main__':
    # manage config
    config = Configuration()
    args = config.get_shell_args_train()
    config.load_config(args.cfg_dir)
    config.overwrite_config_by_shell_args(args)

    # instantiating all modules by non-singleton factory
    model = ModelFactory.get_model(config.model_config)
    paras = torch.load("../../model_paras/param_cifar_10.pt")
    model.load_state_dict(paras)
    model.set_eval()
    dataset = DatasetFactory.get_data_loader(config.dataset_config)
    data_loader = dataset.get_data_loader()

    for step, data in enumerate(data_loader):
        imgs = data[0]
        gt_label = data[1]
        res = model(imgs)
        _, indices = torch.max(res, 1)
        set_to_counter(indices, gt_label)
        print("step=", step)
    for (k1, v1), (k2, v2) in zip(right_prediction_dict.items(), whole_class_num.items()):
        assert k1 == k2
        precision = v1 / v2
        precision_dict[k1] = precision
    plt.plot(list(precision_dict.keys()), list(precision_dict.values()))
    plt.show()

