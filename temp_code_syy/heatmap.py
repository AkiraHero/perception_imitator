'''
1、根据每一帧的检测结果画出该帧下的FP框的Heatmap图
具体实现：需要选取bev下的一定范围（待定），进行固定大小数组的初始化；将每个FP框按照中心点以及长宽生成heatmap填入该矩阵
2、将多帧的数据汇总：简单叠加取平均值，得到最终的heatmap。
'''

from os import name
import os
import numpy as np
import cv2
import pickle
import math
import matplotlib.pyplot as plt
from PIL import Image
from numpy.testing._private.utils import break_cycles

def scale_to_255(a, min, max, dtype=np.uint8):
	return ((a - min) / float(max - min) * 255).astype(dtype)
    
def get_Gausseheat(bbox_length, bbox_width):
    size = int(math.sqrt(bbox_length * bbox_width) * 20) 
    kernel=cv2.getGaussianKernel(size,size/5)
    kernel=kernel*kernel.T
    # scales all the values and make the center vaule of kernel to be 1.0
    kernel=kernel/np.max(kernel)
    # heatmap=kernel*255
    heatmap=kernel
    # heatmap=heatmap.astype(np.uint8)
    # heatmap=cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    # cv2.imshow('heatmap',heatmap)
    # cv2.waitKey(0)
    return size, heatmap

def plot_cloud_bev(pointcloud, res, side_range, fwd_range):    
    x_points = pointcloud[:, 0]
    y_points = pointcloud[:, 1]
    z_points = pointcloud[:, 2]
    
    # 获得区域内的点
    f_filt = np.logical_and(x_points > fwd_range[0], x_points < fwd_range[1])
    s_filt = np.logical_and(y_points > side_range[0], y_points < side_range[1])
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]
    
    x_img = (-y_points / res).astype(np.int32)
    y_img = (-x_points / res).astype(np.int32)
    # 调整坐标原点
    x_img -= int(np.floor(side_range[0]) / res)
    y_img += int(np.floor(fwd_range[1]) / res)
    
    # 填充像素值
    height_range = (-3, 3)
    pixel_value = np.clip(a=z_points, a_max=height_range[1], a_min=height_range[0])

    pixel_value = scale_to_255(pixel_value, height_range[0], height_range[1])
    
    # 创建图像数组
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[y_img, x_img] = pixel_value
    
    # imshow （灰度）
    # im2 = Image.fromarray(im)
    # im2.show()
    
    # imshow （彩色）
    # plt.imshow(im, cmap="nipy_spectral", vmin=0, vmax=255)
    # plt.show()

    return x_max, y_max

def get_heatmaps(img_id, gt_dt_matching_res, result, res, side_range, fwd_range, x_max, y_max):    
    dt_bboxes = result[img_id]['boxes_lidar']   # id帧下的所有检测框
    matching_index = gt_dt_matching_res['bev'][img_id]

    im = np.zeros([y_max, x_max])
    for index in range(len(dt_bboxes)):
        if index not in matching_index: # 处理一个检测框
            x_center = dt_bboxes[index][0]
            y_center = dt_bboxes[index][1]
            bbox_length = dt_bboxes[index][3]
            bbox_width = dt_bboxes[index][4]

            if x_center<fwd_range[0]+3 or x_center>fwd_range[1]-3 or y_center<side_range[0]+3 or y_center>side_range[1]-3:
                break

            size, heatmap = get_Gausseheat(bbox_length, bbox_width) # 

            # 调整坐标到图片坐标系
            img_x_center = (-y_center / res).astype(np.int32)
            img_y_center = (-x_center / res).astype(np.int32)        
            img_x_center -= int(np.floor(side_range[0]) / res)
            img_y_center += int(np.floor(fwd_range[1]) / res)

            # 将gausse heat画在图片上
            im[img_y_center - int(size/2):img_y_center - int(size/2) + size, img_x_center - int(size/2):img_x_center - int(size/2) + size] = heatmap

    # im = cv2.applyColorMap(im, cv2.COLORMAP_HOT)[...,::-1]
    # plt.imshow(im, cmap="nipy_spectral", vmin=0, vmax=255)
    # plt.show()

    return im


if __name__ == '__main__':
    for img_id in range(1,2):
        print("now processing: %06d"%img_id)

        lidar_path = r'F:/Kitti/data_object_velodyne/training/velodyne/%06d.bin' % img_id  ## Path ## need to be changed
        exp_path = 'D:/1Pjlab/Datasets/kitti-pvrcnn-epch8369-392dropout/default'

        # 设置鸟瞰图范围
        res = 0.05 # 分辨率0.05m
        side_range = (-35, 35)  # 左右距离
        fwd_range = (0, 70)  # 后前距离

        # 提取点云数据
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)  # .astype(np.float16)
        x_max, y_max = plot_cloud_bev(points, res, side_range, fwd_range) # 画点云鸟瞰图，并返回图像大小
        
        i = 0
        all_heatmap = np.zeros([y_max, x_max], dtype=np.uint8)  # 汇总heatmap
        for exp_id in os.listdir(exp_path):
            print('\r', exp_id)
            i = i+1
            exp_name = os.path.join(exp_path, exp_id)   # 每次dropout实验的文件夹
            matching_name = os.path.join(exp_name, 'gt_dt_matching_res.pkl')
            result_name = os.path.join(exp_name, 'result.pkl')

            with open(matching_name, 'rb') as f:
                gt_dt_matching_res = pickle.load(f)
            with open(result_name, 'rb') as f:
                result = pickle.load(f)

            one_heatmap = get_heatmaps(img_id, gt_dt_matching_res, result, res, side_range, fwd_range, x_max, y_max)
            all_heatmap = all_heatmap + one_heatmap

            if i == 1:
                break

        # all_heatmap = all_heatmap / len(os.listdir(exp_path)) * 255
        all_heatmap = all_heatmap /  1 * 255
        all_heatmap = all_heatmap.astype(np.uint8)

        all_heatmap = cv2.applyColorMap(all_heatmap, cv2.COLORMAP_HOT)[...,::-1]
        # plt.imshow(all_heatmap, alpha = 0.5,cmap="nipy_spectral", vmin=0, vmax=255)
        # plt.show()
        plt.imshow(all_heatmap, alpha = 1,cmap="nipy_spectral", vmin=0, vmax=255)
        plt.show()


    '''
    exp_path = 'D:/1Pjlab/Datasets/kitti-pvrcnn-epch8369-392dropout/default'
    # 设置鸟瞰图范围
    res = 0.05 # 分辨率0.05m
    side_range = (-35, 35)  # 左右距离
    fwd_range = (0, 70)  # 后前距离
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    all_heatmap = np.zeros([7481, y_max, x_max], dtype=np.float16)  # 汇总heatmap

    i = 0
    for exp_id in os.listdir(exp_path):
        i = i+1
        print('\r', exp_id)
        exp_name = os.path.join(exp_path, exp_id)   # 每次dropout实验的文件夹
        matching_name = os.path.join(exp_name, 'gt_dt_matching_res.pkl')
        result_name = os.path.join(exp_name, 'result.pkl')

        with open(matching_name, 'rb') as f:
            gt_dt_matching_res = pickle.load(f)
        with open(result_name, 'rb') as f:
            result = pickle.load(f)

        for img_id in range(0,100):
            print("now processing: %06d"%img_id)

            lidar_path = r'F:/Kitti/data_object_velodyne/training/velodyne/%06d.bin' % img_id  ## Path ## need to be changed

            # 提取点云数据
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)  # .astype(np.float16)
            x_max, y_max = plot_cloud_bev(points, res, side_range, fwd_range) # 画点云鸟瞰图，并返回图像大小

            one_heatmap = get_heatmaps(img_id, gt_dt_matching_res, result, res, side_range, fwd_range, x_max, y_max)
            
            all_heatmap[img_id] = all_heatmap[img_id] + one_heatmap
            
            # plot_heatmap = all_heatmap[img_id]
            # plot_heatmap = plot_heatmap/1 * 255
            # plot_heatmap = plot_heatmap.astype(np.uint8)
            # print(plot_heatmap.shape)
            # plot_heatmap = cv2.applyColorMap(plot_heatmap, cv2.COLORMAP_HOT)[...,::-1]
            # plt.imshow(plot_heatmap, cmap="nipy_spectral", vmin=0, vmax=255)
            # plt.show()


        if i == 100:
            break

    for i in range(0,100):
        plot_heatmap = all_heatmap[i]
        plot_heatmap = plot_heatmap/100 * 255
        plot_heatmap = plot_heatmap.astype(np.uint8)
        plot_heatmap = cv2.applyColorMap(plot_heatmap, cv2.COLORMAP_HOT)[...,::-1]
        plt.imshow(plot_heatmap, cmap="nipy_spectral", vmin=0, vmax=255)
        plt.show()

    '''




 



