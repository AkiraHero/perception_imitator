import pickle
import torch
import os
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

def get_fp_explicit_dataset_pkl():
    datas = []
    labels = [] 

    cloud_path = "F:/Kitti/data_object_velodyne/training/cloud_in_bbox"
    data_root = "D:/1Pjlab/ModelSimulator/data"

    db_file = os.path.join(data_root, "gt_dt_matching_res.pkl")
    print("Loading bbox data from pkl")
    with open(db_file, 'rb') as f:
        db = pickle.load(f)

    cloud_files = os.listdir(cloud_path)
    for one_dt_cloud in cloud_files:
        print("Processing %s"%one_dt_cloud)
        img_id = int(one_dt_cloud.split('-')[0])
        dt_box_id = int(one_dt_cloud.split('-')[1])
        cls_id = label_str2num(one_dt_cloud.split('-')[2])
        gt = gt_str2num(one_dt_cloud.split('-')[3].split('.')[0])
        bbox_para = db['dt_annos'][img_id]['boxes_lidar'][dt_box_id]
        
        # 加载点云数据
        cloud = np.load(os.path.join(cloud_path, one_dt_cloud))
        point_num = cloud.shape[0]
        refl_u = np.mean(cloud[:, 3])
        refl_sigma = np.var(cloud[:, 3])

        data = bbox_para.tolist() + [cls_id, refl_u, refl_sigma, point_num]
        label = [gt]
        
        datas.append(data)
        labels.append(label)

    fp_explicit_data = {'datas': datas, 'labels':labels}
    print(len(fp_explicit_data['datas']))

    #存储fp bbox数据为pkl文件
    fp_explicit_file = "D:/1Pjlab/ModelSimulator/data/fp_explicit_data.pkl"
    with open(fp_explicit_file, "wb") as f:
        pickle.dump(fp_explicit_data, f)


if __name__ == '__main__':
    get_fp_explicit_dataset_pkl()