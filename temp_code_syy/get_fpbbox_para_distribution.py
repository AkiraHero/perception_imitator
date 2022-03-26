import sys
sys.path.append('/cpfs2/user/sunyiyang/Pjlab/ADModel_Pro')
from os import name
import os
import logging
import numpy as np
import cv2
import pickle
import math
import matplotlib.pyplot as plt
from PIL import Image
from numpy.testing._private.utils import break_cycles
from utils.postprocess import compute_overlaps

def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def get_all_exp_fp():
    data_num = 7481
    exp_path = 'C:/Users/Sunyyyy/Desktop/Study/PJLAB/Experiment/Data/kitti-pvrcnn-epch8369-392dropout/default'

    all_exp_all_img_fpbbox = []     # 存储所有实验下所有图片中的FP检测框信息
    all_exp_all_img_score = []

    for exp_id in os.listdir(exp_path): # 循环遍历每次实验
        print('experience_id:', exp_id)
        exp_name = os.path.join(exp_path, exp_id)   # 每次dropout实验的文件夹
        matching_name = os.path.join(exp_name, 'gt_dt_matching_res.pkl')
        result_name = os.path.join(exp_name, 'result.pkl')
        with open(matching_name, 'rb') as f:
            gt_dt_matching_res = pickle.load(f)
        with open(result_name, 'rb') as f:
            result = pickle.load(f)

        per_exp_all_img_fpbbox = []     # 存储一次实验下所有图片中的FP检测框信息
        per_exp_all_img_score = []

        for img_id in range(0, data_num):
            # print("now processing: %06d"%img_id) 
            dt_bboxes = result[img_id]['boxes_lidar']   # id帧下的所有检测框
            scores = result[img_id]['score']    # id帧下的所有检测框得分
            matching_index = gt_dt_matching_res['bev'][img_id]

            per_exp_per_img_fpbbox = []     # 存储一次实验下一帧图片中的FP检测框信息
            per_exp_per_img_score = []

            for fp_index in range(len(dt_bboxes)):
                if fp_index not in matching_index:     # 处理一个FP检测框
                    per_exp_per_img_fpbbox.append(dt_bboxes[fp_index])
                    per_exp_per_img_score.append(scores[fp_index])

            per_exp_all_img_fpbbox.append(per_exp_per_img_fpbbox)
            per_exp_all_img_score.append(per_exp_per_img_score)
            
        all_exp_all_img_fpbbox.append(per_exp_all_img_fpbbox)
        all_exp_all_img_score.append(per_exp_all_img_score)

    print(len(all_exp_all_img_fpbbox))
    print(len(all_exp_all_img_score))
    save_pkl = {'all_exp_fp': all_exp_all_img_fpbbox, 'all_exp_score': all_exp_all_img_score}
    
    with open("C:/Users/Sunyyyy/Desktop/Study/PJLAB/Experiment/Data/kitti-pvrcnn-epch8369-392dropout/process_all_experiment_fp.pkl", "wb") as f:
        pickle.dump(save_pkl, f)

def get_corners(bbox):
        x, y, l, w, yaw = bbox        
        #x, y, w, l, yaw = self.interpret_kitti_label(bbox)
        
        bev_corners = np.zeros((4, 2), dtype=np.float32)
        # rear left
        bev_corners[0, 0] = x - l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[0, 1] = y - l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        # rear right
        bev_corners[1, 0] = x - l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[1, 1] = y - l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front right
        bev_corners[2, 0] = x + l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[2, 1] = y + l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front left
        bev_corners[3, 0] = x + l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[3, 1] = y + l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        return bev_corners

def compute_matches(gt_boxes,
                    pred_boxes, pred_scores,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.
    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """

    if len(pred_scores) == 0:
        return -1 * np.ones([gt_boxes.shape[0]]), np.array([]), np.array([])

    gt_class_ids = np.ones(len(gt_boxes), dtype=int)
    pred_class_ids = np.ones(len(pred_scores), dtype=int)

    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]

    # Compute IoU overlaps [pred_boxes, gt_boxes]
    overlaps = compute_overlaps(pred_boxes, gt_boxes)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > 0:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps

if __name__ == '__main__':
    # # 预处理获取所有实验中所需的fp数据
    # get_all_exp_fp()

    log_file = "C:/Users/Sunyyyy/Desktop/Study/PJLAB/Code/ADModel_Pro/data/log_fp_distribution.txt"
    logger = create_logger(log_file, rank=0)
    logger.info('**********************Start logging**********************')

    all_exp_fp_name = "C:/Users/Sunyyyy/Desktop/Study/PJLAB/Code/ADModel_Pro/data/process_all_experiment_fp.pkl"
    with open(all_exp_fp_name, 'rb') as f:
        all_exp_fp_file = pickle.load(f)

    all_exp_fp = all_exp_fp_file['all_exp_fp']
    all_exp_score = all_exp_fp_file['all_exp_score']

    proc_fp = [[] for _ in range(len(all_exp_fp[0]))]    # 存储每帧图像经过FP配对后的数据
    proc_score_add = [[] for _ in range(len(all_exp_fp[0]))] # 存储每帧图像每组FP对应的得分之和（每组FP的意思是这些组内的FP IoU较大，相当于匹配上）

    proc_fp_mean = [[] for _ in range(len(all_exp_fp[0]))]
    proc_fp_var = [[] for _ in range(len(all_exp_fp[0]))]
    proc_score = [[] for _ in range(len(all_exp_fp[0]))]

    for exp_id in range(len(all_exp_fp)):   # 每次将之气那汇总的中的FP数据和当前exp_id和中的FP数据
        logger.info('now processing experiment:', exp_id)
        print("exp_id:", exp_id)
        one_exp_fp = all_exp_fp[exp_id]
        one_exp_score = all_exp_score[exp_id]

        for img_id in range(len(one_exp_fp)):
            # print("process img %06d" %img_id)
            one_img_fp_bboxes = one_exp_fp[img_id]
            one_img_fp_score = one_exp_score[img_id]
            pro_fp_list = []
            fp_list = []    # 用于存储该exp下该img的所有fp的焦点坐标

            for i in range(len(proc_fp[img_id])):     # 将之前exp处理好第i个fp组中的第一个取出，作为和新的expFP的配对项
                x = proc_fp[img_id][i][0][0]
                y = proc_fp[img_id][i][0][1]
                l = proc_fp[img_id][i][0][3]
                w = proc_fp[img_id][i][0][4]
                yaw = proc_fp[img_id][i][0][6]

                corners = get_corners([x, y, l, w, yaw])
                pro_fp_list.append(corners)
        
            for i in range(len(one_img_fp_bboxes)):     # 将当前数据转成bev视角的四个角点坐标，用于计算IoU进行匹配
                x = one_img_fp_bboxes[i][0]
                y = one_img_fp_bboxes[i][1]
                l = one_img_fp_bboxes[i][3]
                w = one_img_fp_bboxes[i][4]
                yaw = one_img_fp_bboxes[i][6]

                corners = get_corners([x, y, l, w, yaw])
                fp_list.append(corners)
            
            proc_fp_img = np.array(pro_fp_list)     # 作为配对真值
            fp_list = np.array(fp_list)
            one_img_fp_score = np.array(one_img_fp_score)

            # 将汇总的FP结果与当前exp的结果进行匹配
            gt_match, pred_match, overlaps = compute_matches(proc_fp_img, fp_list, one_img_fp_score, iou_threshold=0.3)

            for i in range(len(pred_match)):
                if pred_match[i] == -1:     # 与之前的FP数据都没有匹配上，那么作为新的FP加入该图片的FP中
                    proc_fp[img_id].append([one_img_fp_bboxes[i]])
                    proc_score_add[img_id].append(one_img_fp_score[i])
                else:       # 配对上了，加入相应的FP组
                    proc_fp[img_id][int(pred_match[i])] = np.vstack((proc_fp[img_id][int(pred_match[i])], one_img_fp_bboxes[i]))
                    proc_score_add[img_id][int(pred_match[i])] += one_img_fp_score[i]

    for img_id in range(len(proc_fp)):
        for fp_id in range(len(proc_fp[img_id])):
            proc_fp_mean[img_id].append(np.mean(proc_fp[img_id][fp_id], axis=0))
            proc_fp_var[img_id].append(np.var(proc_fp[img_id][fp_id], axis=0))

            proc_score[img_id].append(proc_score_add[img_id][fp_id] / len(proc_fp[img_id][fp_id]))

    print(len(proc_fp_mean))
    print(len(proc_fp_var))
    print(len(proc_score))

    save_pkl = {'fp_mean': proc_fp_mean, 'fp_var': proc_fp_var, 'fp_score': proc_score}
    
    with open("C:/Users/Sunyyyy/Desktop/Study/PJLAB/Code/ADModel_Pro/data/fp_distribution.pkl", "wb") as f:
        pickle.dump(save_pkl, f)
