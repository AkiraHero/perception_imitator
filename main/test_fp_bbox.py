import sys
sys.path.append('D:/1Pjlab/ModelSimulator/')
from utils.config.Configuration import Configuration
from factory.model_factory import ModelFactory
from factory.dataset_factory import DatasetFactory
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch
import pandas as pd
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
    args.batch_size = 36625
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

    paras = torch.load("D:/1Pjlab/ModelSimulator/output/fp_gen_model/fp_gen_model890.pt")
    model.generator.load_model_paras(paras)
    model.set_eval()
    dataset = DatasetFactory.get_dataset(config.dataset_config)
    data_loader = dataset.get_data_loader()

    for step, data in enumerate(data_loader):
        dt_box_fp = data
        gen_data = model.generator(dt_box_fp)

        datax = {'x_fp': dt_box_fp[:,0],
                'x_gen':gen_data[0].detach()[:,0]}
        datay = {'y_fp': dt_box_fp[:,1],
                'y_gen':gen_data[0].detach()[:,1]}
        dataz = {'z_fp': dt_box_fp[:,2],
                'z_gen':gen_data[0].detach()[:,2]}
        datal = {'l_fp': dt_box_fp[:,3],
                'l_gen':gen_data[0].detach()[:,3]}                
        dataw = {'w_fp': dt_box_fp[:,4],
                'w_gen':gen_data[0].detach()[:,4]}
        datah = {'h_fp': dt_box_fp[:,5],
                'h_gen':gen_data[0].detach()[:,5]}
        datarot = {'rot_fp': dt_box_fp[:,6],
                'rot_gen':gen_data[0].detach()[:,6]}

        plt.subplot(241)
        labels = 'x_fp','x_gen'#图例
        plt.boxplot([datax['x_fp'], datax['x_gen']],notch=False, widths = 0.5, labels = labels,patch_artist = False, boxprops = {'color':'lightblue','linewidth':'1.0'},
            capprops={'color':'lightblue','linewidth':'1.0'}, whiskerprops={'color':'lightblue','linewidth':'1.0'})
        plt.grid(linestyle="--", alpha=0.3)   
        plt.subplot(242)
        labels = 'y_fp','y_gen'#图例
        plt.boxplot([datay['y_fp'], datay['y_gen']],notch=False, widths = 0.5, labels = labels,patch_artist = False, boxprops = {'color':'lightblue','linewidth':'1.0'},
            capprops={'color':'lightblue','linewidth':'1.0'}, whiskerprops={'color':'lightblue','linewidth':'1.0'})
        plt.grid(linestyle="--", alpha=0.3)   
        plt.subplot(243)
        labels = 'z_fp','z_gen'#图例
        plt.boxplot([dataz['z_fp'], dataz['z_gen']],notch=False, widths = 0.5, labels = labels,patch_artist = False, boxprops = {'color':'lightblue','linewidth':'1.0'},
            capprops={'color':'lightblue','linewidth':'1.0'}, whiskerprops={'color':'lightblue','linewidth':'1.0'})
        plt.grid(linestyle="--", alpha=0.3)   
        plt.subplot(244)
        labels = 'w_fp','w_gen'#图例
        plt.boxplot([dataw['w_fp'], dataw['w_gen']],notch=False, widths = 0.5, labels = labels,patch_artist = False, boxprops = {'color':'lightblue','linewidth':'1.0'},
            capprops={'color':'lightblue','linewidth':'1.0'}, whiskerprops={'color':'lightblue','linewidth':'1.0'})
        plt.grid(linestyle="--", alpha=0.3)   
        plt.subplot(245)
        labels = 'h_fp','h_gen'#图例
        plt.boxplot([datah['h_fp'], datah['h_gen']],notch=False, widths = 0.5, labels = labels,patch_artist = False, boxprops = {'color':'lightblue','linewidth':'1.0'},
            capprops={'color':'lightblue','linewidth':'1.0'}, whiskerprops={'color':'lightblue','linewidth':'1.0'})
        plt.grid(linestyle="--", alpha=0.3)   
        plt.subplot(246)
        labels = 'l_fp','l_gen'#图例
        plt.boxplot([datal['l_fp'], datal['l_gen']],notch=False, widths = 0.5, labels = labels,patch_artist = False, boxprops = {'color':'lightblue','linewidth':'1.0'},
            capprops={'color':'lightblue','linewidth':'1.0'}, whiskerprops={'color':'lightblue','linewidth':'1.0'})
        plt.grid(linestyle="--", alpha=0.3)   
        plt.subplot(247)
        labels = 'rot_fp','rot_gen'#图例
        plt.boxplot([datarot['rot_fp'], datarot['rot_gen']],notch=False, widths = 0.5, labels = labels,patch_artist = False, boxprops = {'color':'lightblue','linewidth':'1.0'},
            capprops={'color':'lightblue','linewidth':'1.0'}, whiskerprops={'color':'lightblue','linewidth':'1.0'})
        plt.grid(linestyle="--", alpha=0.3)   


        plt.show()

        print("step=", step)
    
    pass