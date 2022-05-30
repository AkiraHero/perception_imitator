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

def eval_one_gaussian(index, dataset, data_loader, plot=False):    # eval_one进行单帧结果生成与指标计算
    data = data_loader.dataset[index]
    occupancy = data['occupancy']
    _, label_list, _, _, _ = dataset.get_label(index)

    ######################
    # Add Gaussian Noise #
    ######################
    gaussian_dist = [0, 0.1]    # Gaussian distribution
    fn_rate = 0.5              # Target model's false negative
    guassian_noise_list = dataset.get_gaussian_noise(index, mu=gaussian_dist[0], sigma=gaussian_dist[1], drop=fn_rate)
    fade_score = np.array(len(guassian_noise_list) * [1])

    gt_boxes = np.array(label_list)
    corners = np.array(guassian_noise_list)

    gt_match, pred_match, overlaps = compute_matches(gt_boxes,
                                        corners, fade_score, iou_threshold=0.5)

    num_gt = len(label_list)
    num_pred = len(fade_score)

    if plot == True:
        # Visualization
        plot_bev(occupancy, label_list, window_name='GT')
        plot_bev(occupancy, guassian_noise_list, window_name='Gaussian')

    return num_gt, num_pred, fade_score, pred_match

def eval_one_GM(index, dataset, data_loader, plot=False):    # eval_one进行单帧结果生成与指标计算
    data = data_loader.dataset[index]
    occupancy = data['occupancy']
    _, label_list, _, _, _ = dataset.get_label(index)

    ########################
    # Add Multimodal Noise #
    ########################
    fn_rate = 0.5             # Target model's false negative
    multimodal_noise_list =  dataset.get_multimodal_noise(index, drop=fn_rate)
    fade_score = np.array(len(multimodal_noise_list) * [1])

    gt_boxes = np.array(label_list)
    corners = np.array(multimodal_noise_list)

    gt_match, pred_match, overlaps = compute_matches(gt_boxes,
                                        corners, fade_score, iou_threshold=0.5)

    num_gt = len(label_list)
    num_pred = len(fade_score)

    if plot == True:
        # Visualization
        plot_bev(occupancy, label_list, window_name='GT')
        plot_bev(occupancy, multimodal_noise_list, window_name='MultiModel')

    return num_gt, num_pred, fade_score, pred_match

def eval_dataset_gaussian(dataset, data_loader, e_range='all'):
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
                eval_one_gaussian(image_id, dataset, data_loader, plot=False)
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

def eval_dataset_GM(dataset, data_loader, e_range='all'):
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
                eval_one_GM(image_id, dataset, data_loader, plot=False)
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

    # instantiating loss and dataset
    perception_loss_func = CustomLoss(config.training_config['loss_function'])
    prediction_loss_func = SmoothL1Loss()
    dataset = DatasetFactory.get_dataset(config.dataset_config)
    data_loader = dataset.get_data_loader()

    # get GT
    index = 0

    gaussian_metrics, _, _ = eval_dataset_gaussian(dataset, data_loader)
    print(gaussian_metrics)

    GM_metrics, _, _ = eval_dataset_GM(dataset, data_loader)
    print(GM_metrics)

