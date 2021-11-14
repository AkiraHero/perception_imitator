import os
import pickle
from typing import Counter
import numpy as np

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

def num2class(num):
    if num >= 0 and num < 3:
        clss = 0
    elif num >= 3 and num < 6:
        clss = 1
    elif num >= 6 and num < 9:
        clss = 2
    elif num >= 9:
        clss = 3
    return clss

def hard_num2class(num):
    if num == 0:
        clss = 0
    elif num == 1:
        clss = 1
    elif num == 2:
        clss = 2
    elif num >= 3:
        clss = 3
    return clss

with open("D:/1Pjlab/ADModel_Pro/data/fp_difficult.pkl", 'rb') as f:
    fp_difficult = pickle.load(f)
print(fp_difficult.keys())
print(fp_difficult['image'])
print(fp_difficult['dtbox_id'])
print(fp_difficult['difficult'])
print('-'*10)

with open("D:/1Pjlab/ADModel_Pro/data/gt_dt_matching_res.pkl", 'rb') as f:
    gt_dt = pickle.load(f)

gt_box = gt_dt['gt_annos'][1]['gt_boxes_lidar']
for i in gt_dt['gt_annos'][1]['name'][gt_dt['gt_annos'][1]['name'] != 'DontCare']:
    gt_class = np.array([label_str2num(i) for i in gt_dt['gt_annos'][1]['name'][gt_dt['gt_annos'][1]['name'] != 'DontCare']])[:, np.newaxis]
print(np.concatenate((gt_box, gt_class), axis=1))
input_gt_box = np.concatenate((gt_box, gt_class), axis=1).flatten().tolist()
input_gt_box = input_gt_box[:80] + [0,]*(80-len(input_gt_box))
print(input_gt_box)
print('-'*10)

all_img_fp_difficult = []
for img_id in range(7481):   # 图片序号
    have_fp = 0
    num_all_fp = 0
    num_easy_fp = 0
    num_hard_fp = 0

    dtbox_id_index = [i for i,v in enumerate(fp_difficult['image']) if v==img_id]   # 获取该图中的所有fp对应的索引号
    if len(dtbox_id_index) != 0:
        have_fp = 1
        num_all_fp = len(dtbox_id_index)

        difficult = fp_difficult['difficult'][dtbox_id_index]
        for j in Counter(difficult).keys():
            if j in [1, 2]:
                num_hard_fp = num_hard_fp + Counter(difficult)[j]
            elif j in [0]:
                num_easy_fp = num_easy_fp + Counter(difficult)[j]
    else:
        pass

    # 将个数转为训练所需的对应类别0:0~2个；1:3~5个；2:6~8个；3:大于9个
    num_all_fp = num2class(num_all_fp)
    num_easy_fp = num2class(num_easy_fp)
    num_hard_fp = hard_num2class(num_hard_fp)

    gt_box = gt_dt['gt_annos'][img_id]['gt_boxes_lidar']
    gt_class = np.array([label_str2num(i) for i in gt_dt['gt_annos'][img_id]['name'][gt_dt['gt_annos'][img_id]['name'] != 'DontCare']])[:, np.newaxis]
    input_gt_box = np.concatenate((gt_box, gt_class), axis=1).flatten().tolist()
    input_gt_box = input_gt_box[:80] + [0,]*(80-len(input_gt_box))
    
    img_fp_difficult = {'image': img_id, 'gtbox_input': input_gt_box, 'have_fp': have_fp, 'all_fp': num_all_fp, 'easy_fp': num_easy_fp, 'hard_fp': num_hard_fp}
    all_img_fp_difficult.append(img_fp_difficult)

with open("D:/1Pjlab/ADModel_Pro/data/img_fp_difficult.pkl", "wb") as f:
    pickle.dump(all_img_fp_difficult, f)
