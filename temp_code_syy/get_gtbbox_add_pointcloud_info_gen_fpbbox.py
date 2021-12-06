'''
Data：每张图像描述场景的所有gt的bbox + bbox内点云在六个面上的密度
Label：每张图像内的Hard或者Easy FPbbox

'''

from math import cos
import os
import pickle
import numpy as np
from numpy.core.arrayprint import _leading_trailing
import scipy.linalg as linalg

from point_of_bbox import get_cloud_in_gtbbox

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

# axis为旋转轴,radian为旋转角度（弧度
def rotate_mat(axis, radian):
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian)) 
    return rot_matrix

def get_point_density_on_bbox_surface(gt_bbox, cloud_in_gtbbox):
    num_bbox = len(gt_bbox)
    all_density = []

    for gt_id in range(num_bbox):
        l = gt_bbox[gt_id][3]
        w = gt_bbox[gt_id][4]
        h = gt_bbox[gt_id][5]
        axis = [0,0,1]  # 绕z轴旋转
        theta = - gt_bbox[gt_id][-1]
        rot_matrix = rotate_mat(axis, theta)
        only_point_trans = cloud_in_gtbbox[gt_id][:, 0:3] - gt_bbox[gt_id][0:3] # 平移后的数组

        num_point = len(only_point_trans)
        density = np.array([0]*6)   # 存储点云到前、后、左、右、上、下面的密度
        
        if num_point != 0:
            for i in range(num_point):
                only_point_trans_rot = np.dot(rot_matrix, only_point_trans[i])
                distance = [(l/2 - only_point_trans_rot[0])/l, (only_point_trans_rot[0] + l/2)/l, 
                            (w/2 - only_point_trans_rot[1])/w, (only_point_trans_rot[1] + w/2)/w,
                            (h/2 - only_point_trans_rot[2])/h, (only_point_trans_rot[2] + h/2)/h]    # 分别为距离前、后、左、右、上、下面的相对距离
                density[distance.index(min(distance))] += 1
            normal_density = density / num_point
        else:
            normal_density = density
        all_density.append(normal_density)

    return(all_density)

def get_gtbbox_gen_fpbbox():
    # 读取所有检测框数据
    with open(db_file, 'rb') as f:
        db = pickle.load(f)
    print("3d list长度：", len(db['3d']))   # key"3d"下存储了激光点云的检测框匹配结果和检测框数据
    gt_annos = db['gt_annos']   # 真值
    dt_annos = db['dt_annos']   # pvrcnn的检测结果
    
    # 读取检测框难度数据
    with open(diff_file, 'rb') as f:
        diff = pickle.load(f)

    dataset = []
    
    for img_id in range(0,7481):
        print("now processing: %06d"%img_id)
        # print(gt_annos[img_id])
        # print(dt_annos[img_id])
        
        cloud_in_gtbbox = get_cloud_in_gtbbox(img_id, gt_annos)    # 读取GT框内点云数据
        num_dt = len(dt_annos[img_id]['name'])   # 该帧的检测个数
        gt_bboxes = []
        tp_bboxes = []
        fp_bboxes_all = []
        fp_bboxes_easy = []
        fp_bboxes_hard = []
        difficult = []

        # 得到Data
        gt_bbox = gt_annos[img_id]['gt_boxes_lidar']
        gt_class = np.array([label_str2num(i) for i in gt_annos[img_id]['name'][gt_annos[img_id]['name'] != 'DontCare']])[:, np.newaxis]
        density = get_point_density_on_bbox_surface(gt_bbox, cloud_in_gtbbox)
        gt_bboxes = np.concatenate((gt_bbox, gt_class, density), axis=1).flatten().tolist()
        gt_bboxes = gt_bboxes[:350] + [0,]*(350-len(gt_bboxes))  # kitti一张图片的gtbbox上限为25个，25*14=350

        # 得到TP & FPLabel
        for dt_i in range(num_dt): # 处理第dt_i个检测框数据
            if dt_i not in db['3d'][img_id]:    # FP
                fp_box_lidar = dt_annos[img_id]['boxes_lidar'][dt_i]
                fp_bboxes_all.extend(fp_box_lidar.tolist())

                dtbox_id_index = [i for i,v in enumerate(diff['image']) if v==img_id]
                dt_i_fp_index = [i + dtbox_id_index[0]  for i,v in enumerate(diff['dtbox_id'][dtbox_id_index]) if v==dt_i]
                if dt_i_fp_index == []: # 部分difficult中的框因为点数太少而被筛选掉了
                    continue
        
                fp_difficult = diff['difficult'][dt_i_fp_index]
                difficult.append(fp_difficult.tolist())
                if fp_difficult == 0:
                    fp_bboxes_easy.extend(fp_box_lidar.tolist())
                else:
                    fp_bboxes_hard.extend(fp_box_lidar.tolist())
                        
            else:   # TP
                tp_box_lidar = dt_annos[img_id]['boxes_lidar'][dt_i]
                tp_bboxes.extend(tp_box_lidar.tolist())
        difficult = np.array(difficult).reshape(-1).tolist()
        difficult = difficult[:20] + [0,]*(20-len(difficult))
        tp_bboxes = tp_bboxes[:140] + [0,]*(140-len(tp_bboxes)) # 将每幅图像的TPbbox输出固定为20个
        fp_bboxes_all = fp_bboxes_all[:140] + [0,]*(140-len(fp_bboxes_all)) # 将每幅图像的FPbbox输出固定为20个，即最后为20*7=140维，在训练时补零项不参与梯度回传
        fp_bboxes_easy = fp_bboxes_easy[:140] + [0,]*(140-len(fp_bboxes_easy)) # 将每幅图像的easyFPbbox输出固定为20个，即最后为20*7=140维
        fp_bboxes_hard = fp_bboxes_hard[:70] + [0,]*(70-len(fp_bboxes_hard)) # 将每幅图像的hardFPbbox输出固定为10个，即最后为10*7=70维
        gtbbox_gen_fpbbox = {'gt_bboxes': gt_bboxes, 'tp_bboxes': tp_bboxes, 'fp_bboxes_all': fp_bboxes_all, 'fp_bboxes_easy': fp_bboxes_easy, 'fp_bboxes_hard': fp_bboxes_hard, 'difficult': difficult}
        dataset.append(gtbbox_gen_fpbbox)
    with open("D:/1Pjlab/ADModel_Pro/data/gtbbox_gen_20fpbbox_add_density.pkl", "wb") as f:
        pickle.dump(dataset, f)

if __name__ == '__main__':
    data_root = "D:/1Pjlab/ADModel_Pro/data"
    db_file = os.path.join(data_root, "gt_dt_matching_res.pkl")
    cloud_path = "F:/Kitti/data_object_velodyne/training/cloud_in_bbox"
    diff_file = os.path.join(data_root, "fp_difficult.pkl")

    get_gtbbox_gen_fpbbox()
