import os
import pickle
from typing import Counter
import numpy as np

def get_gtbbox_gen_fpbbox():
    # 读取所有检测框数据
    with open(db_file, 'rb') as f:
        db = pickle.load(f)
    print("3d list长度：", len(db['3d']))   # key"3d"下存储了激光点云的检测框匹配结果和检测框数据

    dt_annos = db['dt_annos']   # pvrcnn的检测结果
    # 读取检测框难度数据
    with open('D:/1Pjlab/ADModel_Pro/data/fp_difficult.pkl', 'rb') as f:
        diff = pickle.load(f)
    
    dataset_easy = []
    dataset_hard = []

    for img_id in range(0,7481):
        print("now processing: %06d"%img_id)
        # print(gt_annos[img_id])
        # print(dt_annos[img_id])

        num_dt = len(dt_annos[img_id]['name'])   # 该帧的检测个数

        # 得到Label
        for dt_i in range(num_dt): # 处理第dt_i个检测框数据
            if dt_i not in db['3d'][img_id]:
                fp_box_lidar = dt_annos[img_id]['boxes_lidar'][dt_i]

                dtbox_id_index = [i for i,v in enumerate(diff['image']) if v==img_id]
                dt_i_fp_index = [i + dtbox_id_index[0]  for i,v in enumerate(diff['dtbox_id'][dtbox_id_index]) if v==dt_i]
                if dt_i_fp_index == []: # 部分difficult中的框因为点数太少而被筛选掉了
                    continue
        
                fp_difficult = diff['difficult'][dt_i_fp_index]
                
                if fp_difficult == 0:
                    dataset_easy.append(fp_box_lidar)
                else:
                    dataset_hard.append(fp_box_lidar)

            else:
                pass

    fp_bbox_data_easy = {'fp_bbox_easy': dataset_easy}
    print(len(fp_bbox_data_easy['fp_bbox_easy']))
    fp_bbox_data_hard = {'fp_bbox_hard': dataset_hard}
    print(len(fp_bbox_data_hard['fp_bbox_hard']))

    with open("D:/1Pjlab/ADModel_Pro/data/fp_bbox_data_easy.pkl", "wb") as f:
        pickle.dump(fp_bbox_data_easy, f)
    with open("D:/1Pjlab/ADModel_Pro/data/fp_bbox_data_hard.pkl", "wb") as f:
        pickle.dump(fp_bbox_data_hard, f)

if __name__ == '__main__':
    data_root = "D:/1Pjlab/ADModel_Pro/data"
    db_file = os.path.join(data_root, "gt_dt_matching_res.pkl")

    get_gtbbox_gen_fpbbox()