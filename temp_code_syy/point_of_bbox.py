# 提取检测框中的所有点云

import numpy as np
import os
import pickle
import math
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from numpy.lib.twodim_base import mask_indices

class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.
        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref
        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]
        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)
        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:     这两个rect/ref 是一个东西？
        right x, down y, front z
        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
        TODO(rqi): do matrix multiplication only once for each projection.
    '''
    def __init__(self, calib_filepath, from_video=False):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3,4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3,4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0,[3,3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0,2]
        self.c_v = self.P[1,2]
        self.f_u = self.P[0,0]
        self.f_v = self.P[1,1]
        self.b_x = self.P[0,3]/(-self.f_u) # relative 
        self.b_y = self.P[1,3]/(-self.f_v)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data
    
    def read_calib_from_video(self, calib_root_dir):
        ''' Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        '''
        data = {}
        cam2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_cam_to_cam.txt'))
        velo2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_velo_to_cam.txt'))
        Tr_velo_to_cam = np.zeros((3,4))
        Tr_velo_to_cam[0:3,0:3] = np.reshape(velo2cam['R'], [3,3])
        Tr_velo_to_cam[:,3] = velo2cam['T']
        data['Tr_velo_to_cam'] = np.reshape(Tr_velo_to_cam, [12])
        data['R0_rect'] = cam2cam['R_rect_00']
        data['P2'] = cam2cam['P_rect_02']
        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian  笛卡尔  就是在后面加了一列 1
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
        return pts_3d_hom
 
    # =========================== 
    # ------- 3d to 3d ---------- 
    # =========================== 
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo) # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref) # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))
    
    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''  # n*3 *  3*3  > n*3  > 3*n
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))
 
    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            这部分是我要的，需要把3D坐标转换到激光雷达坐标
            Output: nx3 points in velodyne coord.
        ''' 
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # =========================== 
    # ------- 3d to 2d ---------- 
    # =========================== 
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P)) # nx3
        pts_2d[:,0] /= pts_2d[:,2]
        pts_2d[:,1] /= pts_2d[:,2]
        return pts_2d[:,0:2]
    
    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # =========================== 
    # ------- 2d to 3d ---------- 
    # =========================== 
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:,0]-self.c_u)*uv_depth[:,2])/self.f_u + self.b_x
        y = ((uv_depth[:,1]-self.c_v)*uv_depth[:,2])/self.f_v + self.b_y
        pts_3d_rect = np.zeros((n,3))
        pts_3d_rect[:,0] = x
        pts_3d_rect[:,1] = y
        pts_3d_rect[:,2] = uv_depth[:,2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)
def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])
def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])
def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])
def transform_from_rot_trans(R, t):
    ''' Transforation matrix from rotation matrix and translation vector. '''
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))
def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr
def scale_to_255(a, min, max, dtype=np.uint8):
	return ((a - min) / float(max - min) * 255).astype(dtype)
def plot_cloud_dev(pointcloud):
    # 点云读取
    print(pointcloud.shape)
    # 设置鸟瞰图范围
    side_range = (-40, 40)  # 左右距离
    fwd_range = (0, 70.4)  # 后前距离
    
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
    
    res = 0.1  # 分辨率0.05m
    x_img = (-y_points / res).astype(np.int32)
    y_img = (-x_points / res).astype(np.int32)
    # 调整坐标原点
    x_img -= int(np.floor(side_range[0]) / res)
    y_img += int(np.floor(fwd_range[1]) / res)
    print(x_img.min(), x_img.max(), y_img.min(), x_img.max())
    
    # 填充像素值
    height_range = (-2, 0.5)
    pixel_value = np.clip(a=z_points, a_max=height_range[1], a_min=height_range[0])

    pixel_value = scale_to_255(pixel_value, height_range[0], height_range[1])
    
    # 创建图像数组
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[y_img, x_img] = pixel_value
    
    # imshow （灰度）
    im2 = Image.fromarray(im)
    im2.show()
    
    # imshow （彩色）
    # plt.imshow(im, cmap="nipy_spectral", vmin=0, vmax=255)
    # plt.show()
def plot_cloud_easy(pointcloud):
    x = pointcloud[:, 0]  # [ 0  3  6  9 12 15 18 21]
    y = pointcloud[:, 1]  # [ 1  4  7 10 13 16 19 22]
    z = pointcloud[:, 2]  # [ 2  5  8 11 14 17 20 23]
    
    # 绘制散点图
    fig = plt.figure()
    ax = Axes3D(fig,auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.scatter(x, y, z)
    
    # 添加坐标轴(顺序是Z, Y, X)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    
    plt.show()
def euclidean_distance(k,h,pointIndex):
    '''
    计算一个点到某条直线的euclidean distance
    :param k: 直线的斜率，float类型
    :param h: 直线的截距，float类型
    :param pointIndex: 一个点的坐标，（横坐标，纵坐标），tuple类型
    :return: 点到直线的euclidean distance，float类型
    '''
    x=pointIndex[0]
    y=pointIndex[1]
    theDistance=math.fabs(h+k*(x-0)-y)/(math.sqrt(k*k+1))
    return theDistance

# 获得img_id图像中所有gtbbxes中的点云
def get_cloud_in_gtbbox(img_id, gt_annos):  
    lidar_path = r'F:/Kitti/data_object_velodyne/training/velodyne/%06d.bin' % img_id  ## Path ## need to be changed
    calib_path = r'F:/Kitti/data_object_velodyne/training/calib/%06d.txt' % img_id
    calibs = Calibration(calib_path)

    point_clouds_in_gtbbox = []
    num_dt = len(gt_annos[img_id]['name'])   # 该帧的检测个数

    # 提取点云数据
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)  # .astype(np.float16)
    # plot_cloud_dev(points)

    for gt_i in range(num_dt): # 处理第gt_i个真值框数据
        if gt_annos[img_id]['name'][gt_i] != 'DontCare':
            w_dis_all = []
            l_dis_all = []
            box_lidar = gt_annos[img_id]['gt_boxes_lidar'][gt_i]   # 数据结构为"x,y,z,l,w,h"
            
            # 高长宽
            x = box_lidar[0]
            y = box_lidar[1]
            z = box_lidar[2]
            l = box_lidar[3]
            w = box_lidar[4]
            h = box_lidar[5] 
            theta = box_lidar[6] 

            # 初步筛选，为增加检测速度
            pass_size = 0.5*math.sqrt(l**2 + w**2)
            x_filt = np.logical_and(
                (points[:,0]>x-pass_size), (points[:,0]<x+pass_size))
            y_filt = np.logical_and(
                (points[:,1]>y-pass_size), (points[:,1]<y+pass_size))
            filt_1 = np.logical_and(x_filt, y_filt)
            temp_object_cloud = points[filt_1, :]


            # 精确筛选，过滤该检测框范围外的激光点，使用xoy平面点到直线距离计算
            for i in range(temp_object_cloud.shape[0]):
                w_dis = euclidean_distance(np.tan(theta), y - np.tan(theta)*x, [temp_object_cloud[i,0], temp_object_cloud[i,1]])
                w_dis_all.append(w_dis)
                l_dis = euclidean_distance(-1/np.tan(theta), y + 1/np.tan(theta)*x, [temp_object_cloud[i,0], temp_object_cloud[i,1]])  
                l_dis_all.append(l_dis)   

            xy_filt2 = np.logical_and(
                (w_dis_all<w/2), (l_dis_all<l/2))
            z_filt = np.logical_and(
                (temp_object_cloud[:,2]>(z-h/2)), (temp_object_cloud[:,2]<(z+h/2)))
            filt_2 = np.logical_and(xy_filt2, z_filt)  # 必须同时成立

            object_cloud = temp_object_cloud[filt_2, :]  # 过滤
            point_clouds_in_gtbbox.append(object_cloud)

    return (point_clouds_in_gtbbox)



def get_kitti_object_cloud_v2():

    save_object_cloud_path = r'F:/Kitti/data_object_velodyne/training/cloud_in_bbox'
    data_root = "D:/1Pjlab/ADModel_Pro/data"
    db_file = os.path.join(data_root, "gt_dt_matching_res.pkl")

    # 读取所有检测框数据
    with open(db_file, 'rb') as f:
        db = pickle.load(f)
    print("3d list长度：", len(db['3d']))   # key"3d"下存储了激光点云的检测框匹配结果和检测框数据
    dt_annos = db['dt_annos']   # pvrcnn的检测结果

    # 测试用
    # num = 1
    # print(db['3d'][num])
    # print(db['dt_annos'][num])
    # print(db['gt_annos'][num])

    for img_id in range(0,7481):
        print("now processing: %06d"%img_id)
        lidar_path = r'F:/Kitti/data_object_velodyne/training/velodyne/%06d.bin' % img_id  ## Path ## need to be changed
        # label_path = r'D:\KITTI\Object\training\label_2\%06d.txt' % img_id  ## Path ## need to be changed
        calib_path = r'F:/Kitti/data_object_velodyne/training/calib/%06d.txt' % img_id
        calibs = Calibration(calib_path)

        num_dt = len(dt_annos[img_id]['name'])   # 该帧的检测个数

        # 提取点云数据
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)  # .astype(np.float16)
        # plot_cloud_dev(points)

        for dt_i in range(num_dt): # 处理第dt_i个检测框数据
            if dt_annos[img_id]['name'][dt_i] != 'DontCare':
                w_dis_all = []
                l_dis_all = []
                box_lidar = dt_annos[img_id]['boxes_lidar'][dt_i]   # 数据结构为"x,y,z,l,w,h"
                
                # 高长宽
                x = box_lidar[0]
                y = box_lidar[1]
                z = box_lidar[2]
                l = box_lidar[3]
                w = box_lidar[4]
                h = box_lidar[5] 
                theta = box_lidar[6] 

                # 初步筛选，为增加检测速度
                pass_size = 0.5*math.sqrt(l**2 + w**2)
                x_filt = np.logical_and(
                    (points[:,0]>x-pass_size), (points[:,0]<x+pass_size))
                y_filt = np.logical_and(
                    (points[:,1]>y-pass_size), (points[:,1]<y+pass_size))
                filt_1 = np.logical_and(x_filt, y_filt)
                temp_object_cloud = points[filt_1, :]


                # 精确筛选，过滤该检测框范围外的激光点，使用xoy平面点到直线距离计算
                for i in range(temp_object_cloud.shape[0]):
                    w_dis = euclidean_distance(np.tan(theta), y - np.tan(theta)*x, [temp_object_cloud[i,0], temp_object_cloud[i,1]])
                    w_dis_all.append(w_dis)
                    l_dis = euclidean_distance(-1/np.tan(theta), y + 1/np.tan(theta)*x, [temp_object_cloud[i,0], temp_object_cloud[i,1]])  
                    l_dis_all.append(l_dis)   

                xy_filt2 = np.logical_and(
                    (w_dis_all<w/2), (l_dis_all<l/2))
                z_filt = np.logical_and(
                    (temp_object_cloud[:,2]>(z-h/2)), (temp_object_cloud[:,2]<(z+h/2)))
                filt_2 = np.logical_and(xy_filt2, z_filt)  # 必须同时成立

                object_cloud = temp_object_cloud[filt_2, :]  # 过滤

                # 转换标签，可以自定义
                if dt_annos[img_id]['name'][dt_i] in ['Car']:
                    adjust_label = 'car'
                elif dt_annos[img_id]['name'][dt_i] in ['Pedestrian']:
                    adjust_label = 'pedestrain'
                elif dt_annos[img_id]['name'][dt_i] in ['Cyclist']:
                    adjust_label = 'cyclist'

                # 判断该检测框是tp还是fp
                if dt_i in db['3d'][img_id]:
                    dt_box_prop = 'tp'
                else:
                    dt_box_prop = 'fp'

                # 只有 1-3 个点的记录或者没有点的记录都不要，其实还可以更加严格
                if object_cloud.shape[0] <= 3:
                    print('filter failed...', img_id, adjust_label, dt_i, dt_box_prop)
                    continue

                # 保存每帧各个检测框的结果
                np.save(save_object_cloud_path+'/%06d-%d-%s-%s' % (img_id, dt_i, adjust_label, dt_box_prop) ,object_cloud)

                # plot_cloud_easy(temp_object_cloud)


if __name__ == '__main__':
    get_kitti_object_cloud_v2()
