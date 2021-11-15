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

    # paras_final = torch.load("/home/xlju/Project/VAE_Mnist2/model_50.pt")
    # ref = model.generator.state_dict()
    # for (k1, v1), (k2, v2) in zip(ref.items(), paras_final.items()):
    #     ref[k1] = v2

    paras = torch.load("D:/1Pjlab/ADModel_Pro/output/fp_gen_cloudpoint/80.pt")
    model.generator.load_model_paras(paras)
    model.set_eval()
    model.set_device("cuda:0")
    dataset = DatasetFactory.get_dataset(config.dataset_config)
    data_loader = dataset.get_data_loader()

    gt_fp_bbox = []
    gen_fp_bbox = []

    with torch.no_grad():
        for step, data in enumerate(data_loader):
            print('Step:', step)
            generate_input = data['cloud_point_1000'].cuda()
            dt_box_fp = data['bbox'].cuda()
            gen_data,_,_ = model.generator(generate_input)
            gt_fp_bbox.extend(dt_box_fp)
            gen_fp_bbox.extend(gen_data)

    gt_fp_bbox = np.array(torch.stack(gt_fp_bbox,0).cpu())
    gen_fp_bbox = np.array(torch.stack(gen_fp_bbox,0).cpu())

    datax = {'x_fp': gt_fp_bbox[:,0],
            'x_gen': gen_fp_bbox[:,0]}
    datay = {'y_fp': gt_fp_bbox[:,1],
            'y_gen': gen_fp_bbox[:,1]}
    dataz = {'z_fp': gt_fp_bbox[:,2],
            'z_gen': gen_fp_bbox[:,2]}
    datal = {'l_fp': gt_fp_bbox[:,3],
            'l_gen': gen_fp_bbox[:,3]}                
    dataw = {'w_fp': gt_fp_bbox[:,4],
            'w_gen': gen_fp_bbox[:,4]}
    datah = {'h_fp': gt_fp_bbox[:,5],
            'h_gen': gen_fp_bbox[:,5]}
    datarot = {'rot_fp': gt_fp_bbox[:,6],
            'rot_gen': gen_fp_bbox[:,6]}

    # 还没有封装画图，直接全部写出
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
    
    pass