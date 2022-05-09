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

def eval_one(model, loss_func, config, loader, image_id, device, plot=False, verbose=False):    # eval_one进行单帧结果生成与指标计算
    data = loader.dataset[image_id]
    
    occupancy = torch.from_numpy(data['occupancy']).unsqueeze(0)
    occlusion = torch.from_numpy(data['occlusion']).unsqueeze(0)
    HDmap = torch.from_numpy(data['HDmap']).permute(2, 0, 1)

    # get input
    input = torch.cat((occupancy, occlusion, HDmap), dim=0).float().to(device)

    # get label
    label_map, label_list, future_waypoints, future_waypoints_st = loader.dataset.get_label(image_id)
    label_map = torch.from_numpy(label_map).permute(2, 0, 1).unsqueeze_(0).to(device)

    # Forward Detection
    pred, features = model(input.unsqueeze(0))
    loss, _, _, _ = loss_func(pred, label_map)
    pred.squeeze_(0)
    features.squeeze_(0)
    cls_pred = pred[0, ...]

    corners, scores = filter_pred(config, pred)
    gt_boxes = np.array(label_list)
    gt_match, pred_match, overlaps = compute_matches(gt_boxes,
                                        corners, scores, iou_threshold=0.5)

    # 根据检测匹配结果获取真值waypoints
    mask = pred_match >= 0
    index = [i for i in pred_match if i >= 0]
    gt_way_points = future_waypoints[index]
    gt_way_points_st = future_waypoints_st[index]

    # Forward Prediction
    if len(corners) == 0:
        ADE = None
        FDE = None
        pass
    else:
        box_centers = np.mean(corners, axis=1)

        center_index = - box_centers / 0.2      # 0.2为resolution
        center_index[:, 0] += pred.shape[-2]
        center_index[:, 1] += pred.shape[-1] / 2
        center_index = np.round(center_index / 4).astype(int)        # 4为input_size/feature_size
        
        center_index = np.swapaxes(center_index, 1, 0)
        center_index[0] = np.clip(center_index[0], 0, features.shape[-2] - 1)
        center_index[1] = np.clip(center_index[1], 0, features.shape[-1] - 1)

        actor_features = features[:, center_index[0], center_index[1]].permute(1, 0)
        pred_way_points_st = model.prediction(actor_features).view(actor_features.shape[0], 6, 2).cpu().numpy()     # 标准化预测结果

        pred_way_points = []
        for i in range(len(pred_way_points_st)):
            pred_way_points.append(pred_way_points_st[i] * 40 + box_centers[i])
        pred_way_points = np.stack(pred_way_points, axis=0)         # lidar坐标系下预测结果

        # 进行ADE和FDE指标计算
        ADE = compute_ADE(gt_way_points, pred_way_points[mask])
        FDE = compute_FDE(gt_way_points, pred_way_points[mask])

    num_gt = len(label_list)
    num_pred = len(scores)

    input_1 = torch.split(input, 1, dim=0)[0]     # [0]为occupancy，[1]为occlusion
    input_np_1 = input_1.cpu().permute(1, 2, 0).numpy()
    input_2 = torch.split(input, 1, dim=0)[1] 
    input_np_2 = input_2.cpu().permute(1, 2, 0).numpy() 
    pred_image = get_bev(input_np_2, corners)

    if plot == True:
        # Visualization
        plot_bev(input_np_2, label_list, window_name='GT')
        plot_bev(input_np_2, corners, window_name='Prediction1')
        plot_bev(input_np_1, corners, window_name='Prediction2')
        plot_label_map(cls_pred.cpu().numpy())

    return num_gt, num_pred, scores, pred_image, pred_match, loss.item(), ADE, FDE

def eval_dataset(config, model, loss_func, loader, device, e_range='all'):
    loss_sum = 0
    ADE_sum = 0
    FDE_sum = 0
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
            num_gt, num_pred, scores, pred_image, pred_match, loss, ADE, FDE= \
                eval_one(model, loss_func, config, loader, image_id, device, plot=False)
            gts += num_gt
            preds += num_pred
            loss_sum += loss
            if ADE == None or FDE == None:
                pass
            else:
                ADE_sum += ADE
                FDE_sum += FDE
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
    metrics['ADE'] = ADE_sum / total_num
    metrics['FDE'] = FDE_sum / total_num

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
    prediction_loss_func = SmoothL1Loss()

    paras = torch.load("C:/Users/Sunyyyy/Desktop/Study/PJLAB/Code/ADModel_Pro/output/baseline_kitti_range/50.pt")
    model.load_model_paras(paras)
    model.set_decode(True)
    model.set_eval()
    model.set_device("cuda:0")

    dataset = DatasetFactory.get_dataset(config.dataset_config)
    data_loader = dataset.get_data_loader()

    with torch.no_grad():
        # Eval one pic
        for id in range(1500,1650):
            num_gt, num_pred, scores, pred_image, pred_match, loss, ADE, FDE= \
            eval_one(model, perception_loss_func, config, data_loader, image_id=id, device="cuda", plot=True)

            TP = (pred_match != -1).sum()
            print("Loss: {:.4f}".format(loss))
            print("Precision: {:.2f}".format(TP/num_pred))
            print("Recall: {:.2f}".format(TP/num_gt))
            
        # # Eval all
        # metrics, precisions, recalls, log_images = eval_dataset(config, model, perception_loss_func, data_loader, device="cuda", e_range='all')
        # print(metrics)
