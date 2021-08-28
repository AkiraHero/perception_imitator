from utils.config.Configuration import Configuration
from model.model_factory import ModelFactory
from dataset.dataset_factory import DatasetFactory
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch

right_prediction_dict_target = OrderedDict()
right_prediction_dict_gen = OrderedDict()
whole_class_num = OrderedDict()
precision_dict_target = OrderedDict()
precision_dict_gen = OrderedDict()

for i in range(10):
    right_prediction_dict_target[i] = 0
    right_prediction_dict_gen[i] = 0
    whole_class_num[i] = 0


def set_to_counter(target_model_out, gen_model_out, gt):
    res_target = target_model_out == gt
    res_gen = gen_model_out == gt

    for i1, i2, j in zip(res_target, res_gen, gt):
        k = j.item()
        if i1:
            right_prediction_dict_target[k] = right_prediction_dict_target[k] + 1
        if i2:
            right_prediction_dict_gen[k] = right_prediction_dict_gen[k] + 1
        whole_class_num[k] = whole_class_num[k] + 1


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
    paras = torch.load("/home/xlju/Project/ModelSimulator/output/gen_model.pt")
    model.generator.load_state_dict(paras)
    model.set_eval()
    dataset = DatasetFactory.get_data_loader(config.dataset_config)
    data_loader = dataset.get_data_loader()

    for step, data in enumerate(data_loader):
        imgs = data[0]
        gt_label = data[1]
        res = model.generator(imgs)[0]
        target_res = model.target_model(imgs)
        _, indices_gen = torch.max(res, 1)
        _, indices_tar = torch.max(target_res, 1)

        set_to_counter(indices_tar, indices_gen, gt_label)
        print("step=", step)

    for (k1, v1), (k2, v2), (k3, v3) in zip(right_prediction_dict_target.items(), right_prediction_dict_gen.items(), whole_class_num.items()):
        assert k1 == k2 == k3
        precision_target = v1 / v3
        precision_gen = v2 / v3
        precision_dict_target[k1] = precision_target
        precision_dict_gen[k1] = precision_gen

    plt.plot(list(precision_dict_target.keys()), list(precision_dict_target.values()), '.-', label="target model precision")
    plt.plot(list(precision_dict_gen.keys()), list(precision_dict_gen.values()), '.-', label="our model precision")
    plt.legend()
    plt.show()

    pass
