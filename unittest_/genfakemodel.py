import pickle
import os

import numpy as np

matching_result = None

with open(
        "/home/xlju/Project/OpenPCDet/output/kitti_models/pv_rcnn/default/eval/epoch_8369/val/default/gt_dt_matching_res.pkl",
        'rb') as f:
    matching_result = pickle.load(f)

outpath = "/home/xlju/data/pvrcnn_fake_model_"

class_map = {
    'Car': 1,
    'Pedestrian': 2,
    'Cyclist': 3
}


def name2num(names):
    return np.array([class_map[i] for i in names])


def sort_box_(i, matching_result_valid):
    mat = np.concatenate([i['boxes_lidar'], name2num(i['name']).reshape(-1, 1)], axis=1)
    new_mat = []
    for i in matching_result_valid:
        if i == -1:
            new_mat.append(np.zeros([1, 8]))
        else:
            new_mat.append(mat[i:i+1])
    return np.concatenate(new_mat, axis=0)


def find_interested_gt_index(gt_annos, class_map):
    indices = []
    labels = gt_annos['name']
    for idx, i in enumerate(labels):
        if i in class_map:
            indices.append(idx)
    return indices

output_dict_list = []
pkl = "kitti_pvrcnn_all.pkl"
for idx, i in enumerate(matching_result['dt_annos']):
    print("processing:", idx)
    name = i['frame_id'] + '.pkl'

    matching_info = {
        'camera': matching_result['camera'][idx],
        'bev': matching_result['bev'][idx],
        '3d': matching_result['3d'][idx]
    }
    i['matching_info'] = matching_info
    gt_annos = matching_result['gt_annos'][idx]
    gt_valid_inx = find_interested_gt_index(gt_annos, class_map)
    matching_result_valid = matching_result['3d'][idx][gt_valid_inx]

    output_dict = {}
    output_dict['gt_valid_inx'] = gt_valid_inx
    output_dict['model_info'] = "pv_rcnn_8369.pth"
    output_dict['frame_id'] = i['frame_id']
    output_dict['ordered_lidar_boxes'] = sort_box_(i, matching_result_valid)
    output_dict['ini_frm_annos'] = gt_annos
    output_dict_list.append(output_dict)
with open(pkl, 'wb') as f:
    pickle.dump(output_dict_list, f)
