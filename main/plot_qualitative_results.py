import os
import sys
sys.path.append(os.getcwd())
import torch
from torch.multiprocessing import Pool
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import random
import pickle
from factory.model_factory import ModelFactory
from factory.dataset_factory import DatasetFactory
from utils.config.Configuration import Configuration
from utils.loss import CustomLoss, SmoothL1Loss
from utils.postprocess import *
from utils.visualize import get_bev, plot_bev, plot_label_map
from collections import OrderedDict
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def eval_one(model, loss_func, config, loader, image_id, device, plot=False, verbose=False):    # eval_one进行单帧结果生成与指标计算
    data = loader.dataset[image_id]
    
    occupancy = torch.from_numpy(data['occupancy']).permute(2, 0, 1)
    occlusion = torch.from_numpy(data['occlusion']).permute(2, 0, 1)
    HDmap = torch.from_numpy(data['HDmap']).permute(2, 0, 1)

    # get input
    input = torch.cat((occupancy, occlusion, HDmap), dim=0).float().to(device)
    # input = torch.cat((occupancy, occlusion), dim=0).float().to(device)

    # get label
    label_map, label_list = loader.dataset.get_only_detection_label(image_id)
    label_map = torch.from_numpy(label_map).permute(2, 0, 1).unsqueeze_(0).to(device)

    # Forward Detection
    pred, features = model(input.unsqueeze(0))
    pred.squeeze_(0)
    features.squeeze_(0)
    cls_pred = pred[0, ...]

    corners, scores = filter_pred(config, pred)
    gt_boxes = np.array(label_list)

    input_1 = torch.split(input, 1, dim=0)[0]     # [0]为occupancy，[1]为occlusion
    input_np_1 = input_1.cpu().permute(1, 2, 0).numpy()
    input_2 = torch.split(input, 1, dim=0)[1] 
    input_np_2 = input_2.cpu().permute(1, 2, 0).numpy() 

    if plot == True:
        # Visualization
        plot_bev(input_np_1, label_list, window_name='GT')
        plot_bev(input_np_2, corners, window_name='Prediction1')
        plot_bev(input_np_1, corners, window_name='Prediction2')
        plot_label_map(cls_pred.cpu().numpy())

    return label_list, corners, input_np_1, input_np_2

if __name__ == '__main__':
    # manage config
    config = Configuration()
    args = config.get_shell_args_train()
    args.for_train = False
    args.shuffle = False

    config.load_config("utils/config/samples/sample_carla")
    config.overwrite_config_by_shell_args(args)

    # instantiating all modules by non-singleton factory
    perception_loss_func = CustomLoss(config.training_config['loss_function'])
    prediction_loss_func = SmoothL1Loss()   

    model_base = ModelFactory.get_model(config.model_config)
    paras = torch.load("./output/carla_cp_baseline.pt")
    model_base.load_model_paras(paras)
    model_base.set_decode(True)
    model_base.set_eval()
    model_base.set_device("cuda:0")


    config.load_config("utils/config/samples/sample_carla_improve")
    config.overwrite_config_by_shell_args(args)
      = ModelFactory.get_model(config.model_config)
    paras = torch.load("./output/carla_cp_final.pt")
    model_improve.load_model_paras(paras)
    model_improve.set_decode(True)
    model_improve.set_eval()
    model_improve.set_device("cuda:0")

    dataset = DatasetFactory.get_dataset(config.dataset_config)
    data_loader = dataset.get_data_loader()

    with torch.no_grad():
        # Eval one pic
        for id in range(len(dataset)):
            dt_list, baseline_list, occupancy, occlussion = \
            eval_one(model_base, perception_loss_func, config, data_loader, image_id=id, device="cuda", plot=False)

            _, imporve_list, _, _= \
            eval_one(model_improve, perception_loss_func, config, data_loader, image_id=id, device="cuda", plot=False)

            gaussian_dist = [0, 0.1]    # Gaussian distribution
            fn_rate = 0.4              # Target model's false negative
            guassian_noise_list = dataset.get_gaussian_noise(id, mu=gaussian_dist[0], sigma=gaussian_dist[1], drop=fn_rate)
            gaussian_list = np.array(guassian_noise_list)
            multimodal_noise_list = dataset.get_multimodal_noise(id, drop=fn_rate)
            GM_list = np.array(multimodal_noise_list)

            _, gt_list = np.array(dataset.get_gt_info(id))

            gt = plot_bev(occupancy, gt_list, window_name='GT')
            dt = plot_bev(occupancy, dt_list, window_name='DT')
            baseline = plot_bev(occupancy, baseline_list, window_name='Baseline')
            imitator = plot_bev(occupancy, imporve_list, window_name='Imitator')
            gaussian = plot_bev(occupancy, gaussian_list, window_name='Gaussian')
            GM = plot_bev(occupancy, GM_list, window_name='GM')
            
            plt.figure(dpi=100, figsize=(60,10))
            plt.xticks([])
            plt.yticks([])

            ax1 = plt.subplot(1,6,1)
            ax1.imshow(gt)
            ax1.set_title('GT',fontsize = 30)
            ax1.axis('off')

            ax2 = plt.subplot(1,6,2)
            ax2.imshow(dt)
            ax2.set_title('DT',fontsize = 30)
            ax2.axis('off')
            
            ax3 = plt.subplot(1,6,3)
            ax3.imshow(gaussian)
            ax3.set_title('Gaussian',fontsize = 30)
            ax3.axis('off')

            ax4 = plt.subplot(1,6,4)
            ax4.imshow(GM)
            ax4.set_title('GM',fontsize = 30)
            ax4.axis('off')

            ax5 = plt.subplot(1,6,5)
            ax5.imshow(baseline)
            ax5.set_title('Baseline',fontsize = 30)
            ax5.axis('off')

            ax6 = plt.subplot(1,6,6)
            ax6.imshow(imitator)
            ax6.set_title('Imitator',fontsize = 30)
            ax6.axis('off')


            plt.savefig('./output/pic/%d.eps'%id)
            plt.show()