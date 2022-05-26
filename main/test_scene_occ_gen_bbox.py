import os
import sys
sys.path.append(os.getcwd())
from utils.config.Configuration import Configuration
from utils.visualize import get_bev, plot_bev, plot_label_map
from factory.model_factory import ModelFactory
from factory.dataset_factory import DatasetFactory
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # manage config
    config = Configuration()
    args = config.get_shell_args_train()
    # args.batch_size = 1024
    args.for_train = False
    args.shuffle = False
    config.load_config(args.cfg_dir)
    config.overwrite_config_by_shell_args(args)

    # instantiating all modules by non-singleton factory
    model = ModelFactory.get_model(config.model_config)

    paras = torch.load("./output/GM_gen_fp/2.pt")
    model.generator.load_model_paras(paras)
    model.set_eval()
    model.set_device("cuda:0")
    dataset = DatasetFactory.get_dataset(config.dataset_config)
    data_loader = dataset.get_data_loader()

    car_mean = np.array([17.45, -6.92, 4.72, 1.93, -1.11])
    car_std = np.array([10.43, 8.26, 0.26, 0.077, 1.61])

    with torch.no_grad():
        for index in range(10,30):
            data = data_loader.dataset[index]
            dataset.load_data_to_gpu(data)

            occupancy = data['occupancy'].permute(2, 0, 1)
            occlusion = data['occlusion'].permute(2, 0, 1)
            label_list = data['label_list']
            
            generator_input = torch.cat((occupancy, occlusion), dim=0).unsqueeze(0)    # 将场景描述共同输入
            GMmodel_out, _ = model.generator(generator_input)
            gen_bev_bbox = GMmodel_out['x_rec'].view(-1, 5).cpu()
            gen_bbox_list = []


            for i in range(gen_bev_bbox.shape[0]):
                lidar_para = dataset.unstandardize(gen_bev_bbox[i], car_mean, car_std)
                corners, reg_target = dataset.get_corners(lidar_para.tolist(), use_distribution=False)

                gen_bbox_list.append(corners)

            plot_bev(data['occupancy'].cpu().numpy(), label_list, window_name='GT')
            plot_bev(data['occupancy'].cpu().numpy(), gen_bbox_list, window_name='GEN')

    pass