import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

def compute_3d_box_lidar(x, y, z, l, w, h, yaw):
    # 计算旋转矩阵
    # R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    R = np.array([[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    # 8个顶点的xyz
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    z_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    # 旋转矩阵点乘(3，8)顶点矩阵
    corners_3d_lidar = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    # 加上location中心点，得出8个顶点旋转后的坐标
    corners_3d_lidar += np.vstack([x,y,z])
    return corners_3d_lidar

def draw_box(ax, vertices, axes=[0, 1, 2], color='black'):
    """
    Draws a bounding 3D box in a pyplot axis.
    
    Parameters
    ----------
    pyplot_axis : Pyplot axis to draw in.
    vertices    : Array 8 box vertices containing x, y, z coordinates.
    axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
    color       : Drawing color. Defaults to `black`.
    """
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
    ]
    for connection in connections:
        ax.plot(*vertices[:, connection], c=color, lw=0.5)

def draw_point_cloud(ax, points, title, axes=[0, 1, 2], point_size=0.2, xlim3d=None, ylim3d=None, zlim3d=None):
    """
    Convenient method for drawing various point cloud projections as a part of frame statistics.
    """
    # 设置xyz三个轴的点云范围
    axes_limits = [
        [0, 40], # X axis range
        [-20, 20], # Y axis range
        [-3, 5]    # Z axis range
    ]
    axes_str = ['X', 'Y', 'Z']
    # 禁止显示背后的网格
    ax.grid(False)
    # 创建散点图[1]:xyz数据集，[2]:点云的大小，[3]:点云的反射率数据,[4]:为灰度显示
    ax.scatter(*np.transpose(points[:, axes]), s=point_size, c=points[:, 3], cmap='gray')
    # 设置画板的标题
    ax.set_title(title)
    # 设置x轴标题
    ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
    # 设置y轴标题
    ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
    if len(axes) > 2:
        # 设置限制角度
        ax.set_xlim3d(*axes_limits[axes[0]])
        ax.set_ylim3d(*axes_limits[axes[1]])
        ax.set_zlim3d(*axes_limits[axes[2]])
        # 将背景颜色设置为RGBA格式，目前的参数以透明显示
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # 设置z轴标题
        ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
    else:
        # 2D限制角度，只有xy轴
        ax.set_xlim(*axes_limits[axes[0]])
        ax.set_ylim(*axes_limits[axes[1]])
    # User specified limits
    if xlim3d!=None:
        ax.set_xlim3d(xlim3d)
    if ylim3d!=None:
        ax.set_ylim3d(ylim3d)
    if zlim3d!=None:
        ax.set_zlim3d(zlim3d)

if __name__ == '__main__':
    data_root = "D:/1Pjlab/ADModel_Pro/data" 
    db_file = os.path.join(data_root, "gt_dt_matching_res.pkl")

    for img_id in range(1, 7481):
        print("now processing: %06d"%img_id)
        lidar_path = r'F:/Kitti/data_object_velodyne/training/velodyne/%06d.bin' % img_id  ## Path ## need to be changed
        point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

        # 读取所有检测框数据
        with open(db_file, 'rb') as f:
            db = pickle.load(f)
        dt_annos = db['dt_annos']   # pvrcnn的检测结果

        with open('D:/1Pjlab/ADModel_Pro/data/fp_difficult.pkl', 'rb') as f:
            fp_difficult = pickle.load(f)

        # 创建画布并且绘制点云图
        fig = plt.figure(figsize=(20, 20))
        # 在画板中添加1*1的网格的第一个子图，为3D图像
        ax = fig.add_subplot(111, projection='3d')
        # 改变绘制图像的视角，即相机的位置，elev为Z轴角度，azim为(x,y)角度
        ax.view_init(60,130)
        # 在画板中画出点云显示数据，point_cloud[::x]x值越大，显示的点越稀疏
        draw_point_cloud(ax, point_cloud[::8], "velo_points")

        # 进行fp_bbox的画制
        dtbox_id_index = [i for i,v in enumerate(fp_difficult['image']) if v==img_id]
        dtbox_id = fp_difficult['dtbox_id'][dtbox_id_index]
        difficult = fp_difficult['difficult'][dtbox_id_index]
        for index, id in enumerate(dtbox_id):
            corners_3d_lidar_box = dt_annos[img_id]['boxes_lidar'][id]
            corners_3d_lidar = compute_3d_box_lidar(corners_3d_lidar_box[0], corners_3d_lidar_box[1], corners_3d_lidar_box[2], corners_3d_lidar_box[3], corners_3d_lidar_box[4], corners_3d_lidar_box[5], corners_3d_lidar_box[6])
            if difficult[index] == 0:
                color = 'green'
            elif difficult[index] == 1:
                color = 'yellow'
            else:
                color = 'red'
            draw_box(ax, corners_3d_lidar, color=color)
        plt.show()
