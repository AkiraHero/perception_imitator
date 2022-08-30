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
import json
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
    # HDmap = torch.from_numpy(data['HDmap']).permute(2, 0, 1)

    # get input
    # input = torch.cat((occupancy, occlusion, HDmap), dim=0).float().to(device)
    input = torch.cat((occupancy, occlusion), dim=0).float().to(device)

    # Forward Detection
    pred, features = model(input.unsqueeze(0))
    pred.squeeze_(0)

    corners, scores = filter_pred(config, pred)
    loader.dataset.get_only_detection_label(image_id)

    if len(corners) == 0:
        return None, None
    one_save = []
    for i, corner in enumerate(corners):
        frame_id, x, y, l, w, yaw = loader.dataset.get_yiming_need(image_id, corner)
        one_corner = {'frame_id':frame_id, 'x':np.float(x), 'y':np.float(y), 'l':np.float(l), 'w':np.float(w), 'yaw':np.float(yaw)}

        one_save.append(one_corner)
        
    return frame_id, one_save

def eval_dataset(config, model, loss_func, loader, device, e_range='all'):
    loss_sum = 0
    total_num = len(loader.dataset)

    img_list = range(total_num)
    if e_range != 'all':
        e_range = min(e_range, len(loader.dataset))
        img_list = random.sample(img_list, e_range)

    save_dict = dict()
    with torch.no_grad():
        for image_id in tqdm(img_list):
            frame_id, one_save = eval_one(model, loss_func, config, loader, image_id, device, plot=False)
            if frame_id == None:
                continue

            save_dict['%s'%frame_id] = one_save
    
    with open("./output/yiming_kitti_thre0.5_pp_baseline.json", "w") as f:
        json.dump(save_dict, f)
            
    return True

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
    perception_loss_func = CustomLoss(config.training_config['loss_function'])
    prediction_loss_func = SmoothL1Loss()   

    paras = torch.load("./output/kitti/baseline_pp/best.pt")
    model.load_model_paras(paras)
    model.set_decode(True)
    model.set_eval()
    model.set_device("cuda:0")

    dataset = DatasetFactory.get_dataset(config.dataset_config)
    data_loader = dataset.get_data_loader()

    with torch.no_grad():
        # Eval all
        eval_dataset(config, model, perception_loss_func, data_loader, device="cuda", e_range='all')
