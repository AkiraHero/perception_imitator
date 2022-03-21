import sys
sys.path.append('D:/1Pjlab/ADModel_Pro/')
import torch
from torch.multiprocessing import Pool
import pandas as pd
import numpy as np
import time
import random
from factory.model_factory import ModelFactory
from factory.dataset_factory import DatasetFactory
from utils.config.Configuration import Configuration
from utils.loss import CustomLoss
from utils.postprocess import filter_pred, compute_matches, compute_ap
from utils.visualize import get_bev, plot_bev, plot_label_map
from collections import OrderedDict
import matplotlib.pyplot as plt

def eval_one(model, loss_func, config, loader, image_id, device, plot=False, verbose=False):    # eval_one进行单帧结果生成与指标计算
    data = loader.dataset[image_id]
    
    occupancy = torch.from_numpy(data['occupancy']).unsqueeze_(0)
    occlusion = torch.from_numpy(data['occlusion']).unsqueeze_(0)

    input = torch.cat((occupancy, occlusion), dim=0).float().to(device)

    label_map, label_list = loader.dataset.get_label(image_id)
    # loader.dataset.reg_target_transform(label_map)    # 归一化
    label_map = torch.from_numpy(label_map).permute(2, 0, 1).unsqueeze_(0).to(device)

    # Forward Pass
    t_start = time.time()
    pred = model(input.unsqueeze(0))
    t_forward = time.time() - t_start

    loss, cls_loss, loc_loss = loss_func(pred, label_map)
    pred.squeeze_(0)
    cls_pred = pred[0, ...]

    if verbose:
        print("Forward pass time", t_forward)

    # Filter Predictions
    t_start = time.time()
    corners, scores = filter_pred(config, pred)
    t_post = time.time() - t_start

    if verbose:
        print("Non max suppression time:", t_post)

    gt_boxes = np.array(label_list)
    gt_match, pred_match, overlaps = compute_matches(gt_boxes,
                                        corners, scores, iou_threshold=0.5)

    num_gt = len(label_list)
    num_pred = len(scores)

    input = torch.split(input, 1, dim=0)[1]     # [0]为occupancy，[1]为occlusion
    input_np = input.cpu().permute(1, 2, 0).numpy()
    pred_image = get_bev(input_np, corners)

    if plot == True:
        # Visualization
        plot_bev(input_np, label_list, window_name='GT')
        plot_bev(input_np, corners, window_name='Prediction')
        plot_label_map(cls_pred.cpu().numpy())

    return num_gt, num_pred, scores, pred_image, pred_match, loss.item(), t_forward, t_post

def eval_dataset(config, model, loss_func, loader, device, e_range='all'):
    t_fwds = 0
    t_post = 0
    loss_sum = 0

    img_list = range(len(loader.dataset))
    if e_range != 'all':
        e_range = min(e_range, len(loader.dataset))
        img_list = random.sample(img_list, e_range)

    log_img_list = random.sample(img_list, 10)

    gts = 0
    preds = 0
    all_scores = []
    all_matches = []
    log_images = []

    with torch.no_grad():
        for image_id in img_list:
            #tic = time.time()
            num_gt, num_pred, scores, pred_image, pred_match, loss, t_forward, t_nms = \
                eval_one(model, loss_func, config, loader, image_id, device, plot=False)
            gts += num_gt
            preds += num_pred
            loss_sum += loss
            all_scores.extend(list(scores))
            all_matches.extend(list(pred_match))

            t_fwds += t_forward
            t_post += t_nms

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
    metrics['loss'] = loss_sum / len(img_list)
    metrics['Forward Pass Time'] = t_fwds / len(img_list)
    metrics['Postprocess Time'] = t_post / len(img_list)

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
    loss_func = CustomLoss(config.training_config['loss_function'])

    paras = torch.load("D:/1Pjlab/ADModel_Pro/output/scene_occ_heatmap/400.pt")
    model.load_model_paras(paras)
    model.set_decode(True)
    model.set_eval()
    model.set_device("cuda:0")

    dataset = DatasetFactory.get_dataset(config.dataset_config)
    data_loader = dataset.get_data_loader()

    with torch.no_grad():
        # Eval one pic
        num_gt, num_pred, scores, pred_image, pred_match, loss, t_forward, t_nms = \
        eval_one(model, loss_func, config.testing_config, data_loader, image_id=4, device="cuda", plot=True)

        TP = (pred_match != -1).sum()
        print("Loss: {:.4f}".format(loss))
        print("Precision: {:.2f}".format(TP/num_pred))
        print("Recall: {:.2f}".format(TP/num_gt))
        print("forward pass time {:.3f}s".format(t_forward))
        print("nms time {:.3f}s".format(t_nms))

        # # Eval all
        # metrics, precisions, recalls, log_images = eval_dataset(config.testing_config, model, loss_func, data_loader, device="cuda", e_range='all')
        # print(metrics)
         