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

def eval_one(index, model, dataset, data_loader, plot=False):    # eval_one进行单帧结果生成与指标计算
    data = data_loader.dataset[index]
    occupancy = data['occupancy']
    _, label_list, _, _, _ = dataset.get_label(index)

    input = torch.from_numpy(data['GT_bbox']).float().cuda()
    if input.shape[0] == 0:
        return 0, 0, [], []
    pred = model(input)

    Sigmoid = torch.nn.Sigmoid()
    cls = Sigmoid(pred[:, 0])
    reg = pred[:, 1:]

    mask = cls > 0.6
    actornoise_box = input[mask] + reg[mask]
    actornoise_list = []

    for i in range(actornoise_box.shape[0]):
        corners, _ = dataset.get_corners(actornoise_box[i].tolist(), use_distribution=False)
        actornoise_list.append(corners)

    gt_boxes = np.array(label_list)
    corners = np.array(actornoise_list)
    fade_score = np.array(len(actornoise_list) * [1])

    gt_match, pred_match, overlaps = compute_matches(gt_boxes,
                                        corners, fade_score, iou_threshold=0.7)

    num_gt = len(label_list)
    num_pred = len(fade_score)

    if plot == True:
        # Visualization
        plot_bev(occupancy, label_list, window_name='GT')
        plot_bev(occupancy, actornoise_list, window_name='Actornoise')

    return num_gt, num_pred, fade_score, pred_match

def eval_dataset(model, dataset, data_loader, e_range='all'):
    total_num = len(dataset)

    img_list = range(total_num)
    if e_range != 'all':
        e_range = min(e_range, len(dataset))
        img_list = random.sample(img_list, e_range)

    gts = 0
    preds = 0
    all_scores = []
    all_matches = []

    with torch.no_grad():
        for image_id in tqdm(img_list):
            #tic = time.time()
            num_gt, num_pred, scores, pred_match = \
                eval_one(image_id, model, dataset, data_loader, plot=False)
            gts += num_gt
            preds += num_pred
            all_scores.extend(list(scores))
            all_matches.extend(list(pred_match))
            
    all_scores = np.array(all_scores)
    all_matches = np.array(all_matches)
    sort_ids = np.argsort(all_scores)
    all_matches = all_matches[sort_ids[::-1]]

    metrics = {}
    AP, precisions, recalls, precision, recall = compute_ap(all_matches, gts, preds)
    metrics['AP'] = AP
    metrics['Precision'] = precision
    metrics['Recall'] = recall

    return metrics, precisions, recalls

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

    paras = torch.load("./output/actor_noise/85.pt")
    model.load_model_paras(paras)
    model.set_eval()
    model.set_device("cuda:0")

    dataset = DatasetFactory.get_dataset(config.dataset_config)
    data_loader = dataset.get_data_loader()

    metrics, _, _ = eval_dataset(model, dataset, data_loader)
    print(metrics)

