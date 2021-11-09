import os
import pickle
from typing import Counter
import numpy as np

def label_str2num(clss):
    d = {
        'car': 1,
        'pedestrian': 2,
        'cyclist': 3
    }
    return d[clss]

def gt_str2num(label):
    d = {
        'tp': 0,
        'fp': 1,
    }
    return d[label]

cloud_path = "F:/Kitti/data_object_velodyne/training/cloud_in_bbox"
data_root = "D:/1Pjlab/ModelSimulator/data"

db_file = os.path.join(data_root, "gt_dt_matching_res.pkl")
print("Loading bbox data from pkl")
with open(db_file, 'rb') as f:
    db = pickle.load(f)

cloud_files = os.listdir(cloud_path)

fp_cloud_ponits = []
for one_dt_cloud in cloud_files:
    print("Processing %s"%one_dt_cloud)
    img_id = int(one_dt_cloud.split('-')[0])
    dt_box_id = int(one_dt_cloud.split('-')[1])
    cls_id = label_str2num(one_dt_cloud.split('-')[2])
    gt = gt_str2num(one_dt_cloud.split('-')[3].split('.')[0])
    bbox_para = db['dt_annos'][img_id]['boxes_lidar'][dt_box_id]

    if 1 == gt:
        # 加载点云数据
        cloud = np.load(os.path.join(cloud_path, one_dt_cloud))
        
        # 将点云个数补零或者筛选至1000个
        if cloud.shape[0] >= 1000:  
            cloud = cloud[:1000,:]
        else:
            cloud=np.pad(cloud,((0,1000-cloud.shape[0]),(0,0)),'constant', constant_values=(0,0)) 

        fp_cloud_point = {'cloud_point_1000': cloud, 'bbox': bbox_para}
        fp_cloud_ponits.append(fp_cloud_point)

print(len(fp_cloud_ponits))

with open("D:/1Pjlab/ModelSimulator/data/fp_cloud_point.pkl", "wb") as f:
    pickle.dump(fp_cloud_ponits, f)
