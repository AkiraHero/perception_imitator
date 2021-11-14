'''
Data：每张图像描述场景的所有gt的bbox
Label：每张图像内的FPbbox

'''

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

def get_gtbbox_gen_fpbbox():
    # 读取所有检测框数据
    with open(db_file, 'rb') as f:
        db = pickle.load(f)
    print("3d list长度：", len(db['3d']))   # key"3d"下存储了激光点云的检测框匹配结果和检测框数据
    gt_annos = db['gt_annos']   # 真值
    dt_annos = db['dt_annos']   # pvrcnn的检测结果

    dataset = []
    
    for img_id in range(0,7481):
        print("now processing: %06d"%img_id)
        # print(gt_annos[img_id])
        # print(dt_annos[img_id])

        num_dt = len(dt_annos[img_id]['name'])   # 该帧的检测个数
        gt_bboxes = []
        fp_bboxes = []

        # 得到Data
        gt_bbox = gt_annos[img_id]['gt_boxes_lidar']
        gt_class = np.array([label_str2num(i) for i in gt_annos[img_id]['name'][gt_annos[img_id]['name'] != 'DontCare']])[:, np.newaxis]
        gt_bboxes = np.concatenate((gt_bbox, gt_class), axis=1).flatten().tolist()
        gt_bboxes = gt_bboxes[:48] + [0,]*(48-len(gt_bboxes))

        # 得到Label
        for dt_i in range(num_dt): # 处理第dt_i个检测框数据
            if dt_i not in db['3d'][img_id]:
                fp_box_lidar = dt_annos[img_id]['boxes_lidar'][dt_i]
                fp_bboxes.extend(fp_box_lidar.tolist())
            else:
                pass

        fp_bboxes = fp_bboxes[:28] + [0,]*(28-len(fp_bboxes)) # 将每幅图像的FPbbox输出固定为4个，即最后为4*7=28维
    
        gtbbox_gen_fpbbox = {'gt_bboxes': gt_bboxes, 'fp_bboxes': fp_bboxes}
        dataset.append(gtbbox_gen_fpbbox)

    with open("D:/1Pjlab/ADModel_Pro/data/gtbbox_gen_fpbbox.pkl", "wb") as f:
        pickle.dump(dataset, f)

if __name__ == '__main__':
    data_root = "D:/1Pjlab/ADModel_Pro/data"
    db_file = os.path.join(data_root, "gt_dt_matching_res.pkl")

    get_gtbbox_gen_fpbbox()
