import sys
sys.path.append('D:/1Pjlab/ADModel_Pro/')
from utils.config.Configuration import Configuration
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

    paras = torch.load("D:/1Pjlab/ADModel_Pro/output/scene_occ_gen_fp_cls/50.pt")
    model.generator.load_model_paras(paras)
    model.set_eval()
    model.set_device("cuda:0")
    dataset = DatasetFactory.get_dataset(config.dataset_config)
    data_loader = dataset.get_data_loader()

    with torch.no_grad():
        for step, data in enumerate(data_loader):
            dataset.load_data_to_gpu(data)

            occupancy = data['occupancy'].unsqueeze(1)
            occlusion = data['occlusion'].unsqueeze(1)
            label_map = data['label_map']
            fp_cls = label_map[..., 0].unsqueeze(1)
            
            generator_input = torch.cat((occupancy, occlusion), dim=1)    # 将场景描述共同输入
            gen_fp_cls, _, _ = model.generator(generator_input)

            plt.imshow(fp_cls.squeeze().cpu().numpy())
            plt.show()
            plt.clf()
            plt.imshow(gen_fp_cls.squeeze().cpu().numpy())
            plt.show()
    pass