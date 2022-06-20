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

    # get label
    label_map, label_list = loader.dataset.get_only_detection_label(image_id)
    label_map = torch.from_numpy(label_map).permute(2, 0, 1).unsqueeze_(0).to(device)

    # Forward Detection
    pred, _, hard_att_mask = model(input.unsqueeze(0))
    # hard_att_mask = (soft_att_mask > 0.2).long()

    loss, _, _, _ = loss_func(pred, label_map, hard_att_mask)
    pred.squeeze_(0)
    hard_att_mask.squeeze_()

    # plt.imshow(hard_att_mask.cpu())
    # plt.show()
    
    # pred[0] = pred[0] * hard_att_mask
    cls_pred = pred[0, ...]

    corners, scores = filter_pred(config, pred)
    gt_boxes = np.array(label_list)
    gt_match, pred_match, overlaps = compute_matches(gt_boxes,
                                        corners, scores, iou_threshold=0.5)

    num_gt = len(label_list)
    num_pred = len(scores)

    input_1 = torch.split(input, 1, dim=0)[0]     # [0]为occupancy，[1]为occlusion
    input_np_1 = input_1.cpu().permute(1, 2, 0).numpy()
    input_2 = torch.split(input, 1, dim=0)[1] 
    input_np_2 = input_2.cpu().permute(1, 2, 0).numpy() 
    pred_image = get_bev(input_np_2, corners)

    if plot == True:
        # Visualization
        plot_bev(input_np_1, label_list, window_name='GT')
        # plot_bev(input_np_2, corners, window_name='Prediction1')
        # plot_bev(input_np_1, corners, window_name='Prediction2')
        # plot_label_map(cls_pred.cpu().numpy())

    return num_gt, num_pred, scores, pred_image, pred_match, loss.item()

def eval_dataset(config, model, loss_func, loader, device, e_range='all'):
    loss_sum = 0
    total_num = len(loader.dataset)

    img_list = range(total_num)
    if e_range != 'all':
        e_range = min(e_range, len(loader.dataset))
        img_list = random.sample(img_list, e_range)

    log_img_list = random.sample(img_list, 5)

    gts = 0
    preds = 0
    all_scores = []
    all_matches = []
    log_images = []

    with torch.no_grad():
        for image_id in tqdm(img_list):
            #tic = time.time()
            num_gt, num_pred, scores, pred_image, pred_match, loss = \
                eval_one(model, loss_func, config, loader, image_id, device, plot=True)
            gts += num_gt
            preds += num_pred
            loss_sum += loss
            all_scores.extend(list(scores))
            all_matches.extend(list(pred_match))

            if image_id in log_img_list:
                log_images.append(pred_image)
            #print(time.time() - tic)
            
    all_scores = np.array(all_scores)
    all_matches = np.array(all_matches)
    sort_ids = np.argsort(all_scores)
    all_matches = all_matches[sort_ids[::-1]]

    metrics = {}
    AP, precisions, recalls, precision, recall = compute_ap(all_matches, gts, preds)
    metrics['AP'] = AP
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['loss'] = loss_sum / total_num

    return metrics, precisions, recalls, log_images

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

    paras = torch.load("./output/baseline_attention/55.pt")
    model.load_model_paras(paras)
    model.set_decode(True)
    model.set_eval()
    model.set_device("cuda:0")

    dataset = DatasetFactory.get_dataset(config.dataset_config)
    data_loader = dataset.get_data_loader()

    with torch.no_grad():
        # # Eval one pic
        # for id in range(0, 1):
        #     num_gt, num_pred, scores, pred_image, pred_match, loss, ADE, FDE= \
        #     eval_one(model, perception_loss_func, config, data_loader, image_id=id, device="cuda", plot=True)

        #     TP = (pred_match != -1).sum()
        #     print("Loss: {:.4f}".format(loss))
        #     print("Precision: {:.2f}".format(TP/num_pred))
        #     print("Recall: {:.2f}".format(TP/num_gt))
            
        # Eval all
        metrics, precisions, recalls, log_images = eval_dataset(config, model, perception_loss_func, data_loader, device="cuda", e_range='all')
        print(metrics)
