'''
1、根据每一帧的检测结果画出该帧下的FP框的Heatmap图
具体实现:需要选取bev下的一定范围（待定），进行固定大小数组的初始化；将每个FP框按照中心点以及长宽生成heatmap填入该矩阵
2、将多帧的数据汇总:简单叠加取平均值，得到最终的heatmap。
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

def label_str2num(clss):
    d = {
        'Car': 1,
        'Van': 1,
        'Truck': 1,
        'Tram': 1,
        'Misc': 1,
        'Pedestrian': 2,
        'Person_sitting':2,
        'Cyclist': 3
    }
    return d[clss]

def scale_to_255(a, min, max, dtype=np.uint8):
	return ((a - min) / float(max - min) * 255).astype(dtype)

def gaussian_radius(det_size, min_overlap=0.5):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

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
    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)
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
    print(result[img_id])
    scores = result[img_id]['score']    # id帧下的所有检测框得分
    classes = result[img_id]['name']
    matching_index = gt_dt_matching_res['bev'][img_id]

    im = np.zeros([y_max, x_max])
    for index in range(len(dt_bboxes)):
        if index not in matching_index: # 处理一个检测框
            if scores[index] >= 0.5 and label_str2num(classes[index]) == 1:
                x_center = dt_bboxes[index][0]
                y_center = dt_bboxes[index][1]
                bbox_length = dt_bboxes[index][3]
                bbox_width = dt_bboxes[index][4]
                pixel_length = bbox_length/res
                pixel_width = bbox_width/res

                size = int(gaussian_radius((pixel_length, pixel_width))) * 6

                if x_center-size*res/2<fwd_range[0] or x_center+size*res/2>fwd_range[1] or y_center-size*res/2<side_range[0]+5 or y_center+size*res/2>side_range[1]-5:
                    break

                # 调整坐标到图片坐标系
                img_x_center = (-y_center / res).astype(np.int32)
                img_y_center = (-x_center / res).astype(np.int32)        
                img_x_center -= int(np.floor(side_range[0]) / res)
                img_y_center += int(np.floor(fwd_range[1]) / res)

                # 将gausse heat画在图片上
                draw_umich_gaussian(im, (img_x_center, img_y_center), size)
            else:
                pass

    # im = cv2.applyColorMap(im, cv2.COLORMAP_HOT)[...,::-1]
    # plt.imshow(im, cmap="nipy_spectral", vmin=0, vmax=255)
    # plt.show()

    return im


if __name__ == '__main__':
    data_num = 2
    
    exp_path = 'D:/1Pjlab/Datasets/kitti-pvrcnn-epch8369-392dropout/default'
    # 设置鸟瞰图范围
    res = 0.2 # 分辨率0.2m
    side_range = (-40, 40)  # 雷达坐标系y轴——左右距离
    fwd_range = (0, 70.4)  # 雷达坐标系x轴——后前距离
    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)
    all_heatmap = np.zeros([7481, y_max, x_max], dtype=np.float16)  # 汇总heatmap

    i = 0
    for exp_id in os.listdir(exp_path):

        print('\r', exp_id)
        exp_name = os.path.join(exp_path, exp_id)   # 每次dropout实验的文件夹
        matching_name = os.path.join(exp_name, 'gt_dt_matching_res.pkl')
        result_name = os.path.join(exp_name, 'result.pkl')

        with open(matching_name, 'rb') as f:
            gt_dt_matching_res = pickle.load(f)
        with open(result_name, 'rb') as f:
            result = pickle.load(f)

        for img_id in range(0, data_num):
            # print("now processing: %06d"%img_id)
            one_heatmap = get_heatmaps(img_id, gt_dt_matching_res, result, res, side_range, fwd_range, x_max, y_max)
            
            all_heatmap[img_id] = all_heatmap[img_id] + one_heatmap

        # i = i + 1   
        # if i == 5:
        #     break

    gtheatmap = []
    for i in range(0, data_num):
        plot_heatmap = all_heatmap[i]
        plot_heatmap = plot_heatmap/len(os.listdir(exp_path)) * 255
        # plot_heatmap = plot_heatmap/5 * 255
        plot_heatmap = plot_heatmap.astype(np.uint8)

        plt.clf()

        # lidar_path = r'F:/Kitti/data_object_velodyne/training/velodyne/%06d.bin' % i  ## Path ## need to be changed
        # points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)  # .astype(np.float16)
        # plot_cloud_bev(points, res, side_range, fwd_range)

        # plt.imshow(plot_heatmap, alpha = 1, cmap="nipy_spectral", vmin=0, vmax=255)
        # plt.axis('off')
        # plt.savefig("./output/plot_heatmap/%06d.png" %(i), bbox_inches='tight', pad_inches=0.0)

        heatmap = {'image_id': '%06d'%i, 'gt_heatmap': plot_heatmap}
        gtheatmap.append(heatmap)
        
    # with open("D:/1Pjlab/ADModel_Pro/data/GTheatmap_TP_only_car.pkl", "wb") as f:
    #     pickle.dump(gtheatmap, f)




 



