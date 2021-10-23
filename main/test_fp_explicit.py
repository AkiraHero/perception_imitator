import sys

from numpy.lib.type_check import imag
sys.path.append('D:/1Pjlab/ModelSimulator/')
from utils.config.Configuration import Configuration
from factory.model_factory import ModelFactory
from factory.dataset_factory import DatasetFactory
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch
import numpy as np
import pickle

right_prediction_dict_target = OrderedDict()
right_prediction_dict_gen = OrderedDict()
whole_class_num = OrderedDict()
precision_dict_target = OrderedDict()
precision_dict_gen = OrderedDict()

def change_data_form(data):
    for k, v in data.items():
        if k in ['data']:
            v = torch.stack(v, 0)
            v = v.transpose(0,1).to(torch.float32)
            data[k] = v
        elif k in ['label']:
            v = v[0]
            data[k] = v

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

    paras = torch.load("D:/1Pjlab/ModelSimulator/output/fp_explicit_model/2048-2048-1024-2/310.pt")
    # paras = torch.load("D:/1Pjlab/ModelSimulator/output/fp_explicit_model/1024-2048-2048-2/310.pt")
    model.load_model_paras(paras)
    model.set_eval()
    dataset = DatasetFactory.get_dataset(config.dataset_config)
    data_loader = dataset.get_data_loader()

    accuracy_add = 0
    for step, data in enumerate(data_loader):
        change_data_form(data)

        explicit_data = data['data']
        label = data['label']

        pred = model(explicit_data)
        _, indices = torch.max(pred, 1)

        tag = np.array(label.numpy(), dtype= bool)
        # 保存所有的fp检测结果
        image_fp = data['image'].numpy()[tag]
        id_fp = data['dtbox_id'].numpy()[tag]
        pred_fp = indices[tag].numpy()

        fp_clss_result = {'image': image_fp, "dtbox_id": id_fp, "cls_result": pred_fp}
        print(len(fp_clss_result['image']))

        #存储数据为pkl文件
        fp_clss_file = "D:/1Pjlab/ModelSimulator/output/fp_diff/fp_clss_2.pkl"
        with open(fp_clss_file, "wb") as f:
            pickle.dump(fp_clss_result, f)

        accuracy = np.array((indices == label)).tolist().count(True)/ len(np.array(label).tolist())
        accuracy_add = accuracy_add + accuracy

        print("accuracy =", accuracy)
        print("step =", step, )

    all_accuracy = accuracy_add/(step + 1)
    print("all_accuracy =", all_accuracy)   # 总共63532个数据(tp+fp)

    pass
