import pickle
import torch
import os
import numpy as np

data_root = "D:/1Pjlab/ModelSimulator/data"
db_file = os.path.join(data_root, "gt_dt_matching_res.pkl")
fp_bbox_file = os.path.join(data_root, "fp_bbox_data.pkl")

all_fp_bbox = []

print("Loading bbox data from pkl")
with open(db_file, 'rb') as f:
    db = pickle.load(f)

print(db.keys())
print("3d list长度：", len(db['3d']))
print("倒数第二张图片的gt与dt匹配的索引list：", db['3d'][0])
# print("gt中的标注信息：\n", db['gt_annos'][0]['gt_boxes_lidar'])
# print("dt中的检测框信息：\n", db['dt_annos'][0]['boxes_lidar'])
print("gt中的标注信息：\n", db['gt_annos'][0])
print("dt中的检测框信息：\n", db['dt_annos'][0])

# 将fp bbox筛选出来
for i, one_pic_det in enumerate(db['dt_annos']):
    for j, dt_box in enumerate(one_pic_det['boxes_lidar']):
        if j not in db['3d'][i]:
            all_fp_bbox.append(dt_box)
fp_bbox_data = {'fp_bbox': all_fp_bbox}
print(len(fp_bbox_data['fp_bbox']))

#存储fp bbox数据为pkl文件
with open(fp_bbox_file, "wb") as f:
    pickle.dump(fp_bbox_data, f)