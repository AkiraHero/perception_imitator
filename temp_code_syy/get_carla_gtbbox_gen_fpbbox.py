'''
Data：每张图像描述场景的所有gt的bbox
Label：每张图像内的FPbbox

'''

import os
import pickle
from typing import Counter
import numpy as np
import math

def label_str2num(clss):
    d = {
        'Car': 1,
        'Pedestrian': 2,
        'Cyclist': 3
    }
    return d[clss] if clss in ['Car', 'Pedestrian', 'Cyclist'] else 4

def get_gt_bbox(carla_line):
    cls = label_str2num(carla_line.split(" ")[0])
    [h, w, l] = map(float, carla_line.split(" ")[8:11])
    [y, z, x] = map(float, carla_line.split(" ")[11:14])
    rot = math.atan(math.tan(float(carla_line.split(" ")[14])))

    gt_bbox = [x, y, z, l, w, h, rot, cls]  # 一个框的参数

    return gt_bbox

def get_dt_bbox(carla_line):
    [h, w, l] = map(float, carla_line.split(" ")[8:11])
    [y, z, x] = map(float, carla_line.split(" ")[11:14])
    rot = math.atan(math.tan(float(carla_line.split(" ")[14])))

    dt_bbox = [x, y, z, l, w, h, rot]  # 一个框的参数

    return dt_bbox

def get_gtbbox_gen_fpbbox():
    # 读取所有场景id name
    filelist = os.listdir(dt_file)
    dataset = []
    for file in filelist:
        print("Now processing file:", file)

        f_gt = open(os.path.join(gt_file, file))
        f_dt = open(os.path.join(dt_file, file))
        f_dt_type = open(os.path.join(dt_type_file, file))
        gt_bboxes = [] 
        dt_bboxes = []

        # 得到场景中的gtbboxes
        for line in f_gt.readlines():            
            gt_bbox = get_gt_bbox(line)
            gt_bboxes.extend(gt_bbox)
        gt_bboxes = gt_bboxes[:640] + [0,]*(640-len(gt_bboxes))  # 仿照kitti，将gtbbox上限为80个，80*8=640

        # 得到场景中的fpbboxes对应的index
        fp_index = []
        fp_type = f_dt_type.readlines()
        fp_index1 = fp_type[29].replace('dt_box_matched_idx:  [', '').replace(']', '').strip('\n').split(' ')
        fp_index1 = [int(x) for x in fp_index1 if x!='']
        fp_index2 = fp_type[59].replace('dt_box_matched_idx:  [', '').replace(']', '').strip('\n').split(' ')
        fp_index2 = [int(x) for x in fp_index2 if x!='']

        for i in range(len(fp_index1)):
            if fp_index1[i] == -1 and fp_index2[i] == -1:
                fp_index.append(i)
            else:
                pass

        # 得到场景中的fpbboxes
        for i, line in enumerate(f_dt.readlines()):
            if i in fp_index:
                dt_bbox = get_dt_bbox(line)
                dt_bboxes.extend(dt_bbox)
        dt_bboxes = dt_bboxes[:140] + [0,]*(140-len(dt_bboxes))
    
        gtbbox_gen_fpbbox = {'gt_bboxes': gt_bboxes, 'fp_bboxes_all': dt_bboxes}
        dataset.append(gtbbox_gen_fpbbox)

    with open("D:/1Pjlab/ADModel_Pro/data/carla_gtbbox_gen_20fpbbox.pkl", "wb") as f:
        pickle.dump(dataset, f)

if __name__ == '__main__':
    data_root = "D:/1Pjlab/Datasets/carla_openpcd_data"
    gt_file = os.path.join(data_root, "gt_label/kitti_label")
    dt_file = os.path.join(data_root, "dt")
    dt_type_file = os.path.join(data_root, "dt_type")

    get_gtbbox_gen_fpbbox()
