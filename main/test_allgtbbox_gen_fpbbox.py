from math import trunc
import sys
sys.path.append('D:/1Pjlab/ADModel_Pro/')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import pickle
import pandas as pd
import numpy as np
from utils.config.Configuration import Configuration
from utils.IoU_calculate import boxes_iou3d_cpu
from factory.model_factory import ModelFactory
from factory.dataset_factory import DatasetFactory
from collections import OrderedDict

from temp_code_syy.plot_fpbox_on_point_cloud import *

def list_of_tensor_2_tensor(data):
    data = torch.stack(data, 0)
    data = data.cpu()

    return data

def change_data_form(data):
    for k, v in data.items():
        if k in ['gt_bboxes', 'tp_bboxes', 'fp_bboxes_all', 'fp_bboxes_easy', 'fp_bboxes_hard', 'difficult']:
            v = torch.stack(v, 0)
            v = v.transpose(0,1).to(torch.float32)
            data[k] = v
        else:
            data[k] = v

def plot_all_fp_bboxes(img_id, gt_fp_bboxes, gen_fp_bboxes):
    # print("now processing: %06d"%img_id)
    # lidar_path = r'F:/Kitti/data_object_velodyne/training/velodyne/%06d.bin' % img_id  ## Path ## need to be changed
    # point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    
    # 创建画布并且绘制点云图
    fig = plt.figure(figsize=(20, 20))
    # 在画板中添加1*1的网格的第一个子图，为3D图像
    ax = fig.add_subplot(111, projection='3d')
    # 改变绘制图像的视角，即相机的位置，elev为Z轴角度，azim为(x,y)角度
    ax.view_init(60,130)
    # 在画板中画出点云显示数据，point_cloud[::x]x值越大，显示的点越稀疏
    # draw_point_cloud(ax, point_cloud[::2], "velo_points")

    for _, corners_3d_lidar_box in enumerate(gt_fp_bboxes):
        if torch.abs(corners_3d_lidar_box).sum(dim = 0) > 0: 
            corners_3d_lidar = compute_3d_box_lidar(corners_3d_lidar_box[0], corners_3d_lidar_box[1], corners_3d_lidar_box[2], corners_3d_lidar_box[3], corners_3d_lidar_box[4], corners_3d_lidar_box[5], corners_3d_lidar_box[6])
            color = 'red'
            draw_box(ax, corners_3d_lidar, color=color)
        else:
            pass

    for _, corners_3d_lidar_box in enumerate(gen_fp_bboxes):
        corners_3d_lidar = compute_3d_box_lidar(corners_3d_lidar_box[0], corners_3d_lidar_box[1], corners_3d_lidar_box[2], corners_3d_lidar_box[3], corners_3d_lidar_box[4], corners_3d_lidar_box[5], corners_3d_lidar_box[6])
        color = 'green'
        draw_box(ax, corners_3d_lidar, color=color)

    plt.show()


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

    paras = torch.load("D:/1Pjlab/ADModel_Pro/output/carla_gtbbox_gen_20fpbbox_model/80.pt") # 500、700尚可
    model.generator.load_model_paras(paras)
    model.set_eval()
    model.set_device("cuda:0")
    dataset = DatasetFactory.get_dataset(config.dataset_config)
    data_loader = dataset.get_data_loader()

    gen_input = []
    gt_fp_bbox = []
    gen_fp_bbox = []
    gt_fp_difficult= []

    with torch.no_grad():
        for step, data in enumerate(data_loader):
            change_data_form(data)

            print('Step:', step)
            generate_input = data['gt_bboxes'].cuda()
            dt_box_fp = data['fp_bboxes_all'].cuda()
            # difficult = data['difficult'].cuda()

            gen_data,_,_ = model.generator(generate_input)
            dt_box_fp = dt_box_fp.view(dt_box_fp.shape[0], -1, 7)
            gen_data = gen_data.view(dt_box_fp.shape[0], -1, 7)

            gen_input.extend(generate_input)
            gt_fp_bbox.extend(dt_box_fp)
            gen_fp_bbox.extend(gen_data)
            # gt_fp_difficult.extend(difficult)
    
    gen_input = list_of_tensor_2_tensor(gen_input)
    gt_fp_bbox = list_of_tensor_2_tensor(gt_fp_bbox)
    gen_fp_bbox = list_of_tensor_2_tensor(gen_fp_bbox)

    for i in range(5, 7):
        # print(gen_input[i])
        # print(gt_fp_bbox[i])
        # print(gen_fp_bbox[i])
        plot_all_fp_bboxes(5984 + i, gt_fp_bbox[i], gen_fp_bbox[i])    # 传入图片数，gt_fp_bbox，gen_fp_bbox，并画图

    # all_iou = []
    # for i in range(gen_fp_bbox.shape[0]):
    # for i in range(0,1):
    #     print(gen_fp_bbox[i], gt_fp_bbox[i])
    #     iou_3d = boxes_iou3d_cpu(gen_fp_bbox[i], gt_fp_bbox[i])
    #     all_iou.append(iou_3d)

    # all_iou = list_of_tensor_2_tensor(all_iou)
    # print("IoU_3D is:", all_iou)    # 目前高度的生成可能存在问题，导致3D检测框的IoU值为0，但是从bev视角看，是可以生成有重合部分的FP框的
    # print("difficult:", gt_fp_difficult[0:10])

    pass