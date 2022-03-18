from csv import list_dialects
import math
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
from numpy.core.arrayprint import _leading_trailing
import scipy.linalg as linalg

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

def two_points_2_line(p1, p2):
    k = (p1[1] - p2[1]) / (p1[0] - p2[0])
    b = p2[1] - k * p2[0]
    return k, b


if __name__ == '__main__':
    data_root = "D:/1Pjlab/ADModel_Pro/data"
    db_file = os.path.join(data_root, "gt_dt_matching_res.pkl")

    with open(db_file, 'rb') as f:
        db = pickle.load(f)
    gt_annos = db['gt_annos']   # 真值

    save = []   # 用于存储最终的pkl结果

    for id in range(10,11):
        print("now processing: %06d"%id)

        # 转化后图片信息
        res = 0.2 # 分辨率0.2m
        side_range = (-40, 40)  # 雷达坐标系y轴——左右距离
        fwd_range = (0, 70.4)  # 雷达坐标系x轴——后前距离
        x_max = int((side_range[1] - side_range[0]) / res)
        y_max = int((fwd_range[1] - fwd_range[0]) / res)
        # 数据信息
        img_cam_pos = [int(x_max/2), y_max]  # 相机在图片坐标系中的坐标
        gt_bboxes = gt_annos[id]['gt_boxes_lidar']    # 获取当前Lidar id下的gt检测框
        scene_occupy = np.zeros([y_max, x_max]).astype('int32')
        scene_occlusion = np.ones([y_max, x_max]).astype('int32')

        '''
        Step 1:
        先获取GT物体在地图上的Occupancy
        '''
        # 处理每一个gt_bbox
        for one_gtbbox in gt_bboxes:
            # 高长宽
            lidar_x = one_gtbbox[0]
            lidar_y = one_gtbbox[1]
            lidar_l = one_gtbbox[3]
            lidar_w = one_gtbbox[4]
            theta = one_gtbbox[6]

            # 调整坐标到图片坐标系
            img_x = (-lidar_y / res).astype(np.int32) - int(np.floor(side_range[0]) / res)
            img_y = (-lidar_x / res).astype(np.int32) + int(np.floor(fwd_range[1]) / res)
            img_l = lidar_l / res
            img_w = lidar_w / res

            # 初步筛选，减小计算算量
            pass_size = int(np.ceil(0.5*math.sqrt(img_l**2 + img_w**2)))

            pix_x_min = max(0, img_x - pass_size)
            pix_x_max = min(x_max, img_x + pass_size)
            pix_y_min = max(0, img_y - pass_size)
            pix_y_max = min(y_max, img_y + pass_size)

            for pix_y in range(pix_y_min, pix_y_max):
                for pix_x in range(pix_x_min, pix_x_max):
                    w_dis = euclidean_distance(-np.tan(theta + math.pi/2), img_y + np.tan(theta + math.pi/2)*img_x, [pix_x, pix_y])
                    l_dis = euclidean_distance(1/np.tan(theta + math.pi/2), img_y - 1/np.tan(theta + math.pi/2)*img_x, [pix_x, pix_y])  

                    if w_dis <= img_w / 2 and l_dis <= img_l / 2:
                        scene_occupy[pix_y, pix_x] = 1  
                        
                        '''
                        Step 2:
                        再获取GT物体产生的Occlusion
                        '''
                        point = [pix_x, pix_y]  # 使用这个occupy点进行occupation的计算
                        if pix_x == img_cam_pos[0]:
                            for j in range(0, pix_y):
                                scene_occlusion[j][pix_x] = 0
                                
                        elif pix_x < img_cam_pos[0]:
                            k, b = two_points_2_line(point, img_cam_pos)    # 获取两个点所构成的直线，并判断线段中是否有有Occupancy的点
                            # 从两个方向分别进行采样，防止像素化后的稀疏
                            for j in range(0, pix_y):
                                i = int(np.floor((j - b) / k))
                                if i >= 0 and i < x_max:
                                    scene_occlusion[j][i] = 0
                                i = int(np.ceil((j - b) / k))
                                if i >= 0 and i < x_max:
                                    scene_occlusion[j][i] = 0
                            for i in range(0, pix_x):    
                                j = int(np.floor(k * i + b))
                                if j >= 0 and j < y_max:
                                    scene_occlusion[j][i] = 0
                                j = int(np.ceil(k * i + b))
                                if j >= 0 and j < y_max:
                                    scene_occlusion[j][i] = 0
                        elif pix_x > img_cam_pos[0]: 
                            k, b = two_points_2_line(point, img_cam_pos)    # 获取两个点所构成的直线，并判断线段中是否有有Occupancy的点
                            # 从两个方向分别进行采样，防止像素化后的稀疏
                            for j in range(0, pix_y):
                                i = int(np.floor((j - b) / k))
                                if i >= 0 and i < x_max:
                                    scene_occlusion[j][i] = 0
                                i = int(np.ceil((j - b) / k))
                                if i >= 0 and i < x_max:
                                    scene_occlusion[j][i] = 0
                            for i in range(pix_x, x_max):    
                                j = int(np.floor(k * i + b))
                                if j >= 0 and j < y_max:
                                    scene_occlusion[j][i] = 0
                                j = int(np.ceil(k * i + b))
                                if j >= 0 and j < y_max:
                                    scene_occlusion[j][i] = 0
        
        # 画出Occupancy图
        plt.imshow(scene_occupy, cmap=plt.cm.gray)
        plt.show()

        # '''
        # Step 2:
        # 再获取GT物体产生的Occlusion
        # '''
        for pix_y in range(y_max - 1):  # 减1不考虑相机位置所在行
            for pix_x in range(x_max):
                point = [pix_x, pix_y]  # 判断这个点是否被遮挡
                occl_tag = 0            # 遮挡标记为，0为未被遮挡，1为被遮挡
                    
                if pix_x == img_cam_pos[0]:
                    for j in range(pix_y, img_cam_pos[1]):
                        if scene_occupy[j][pix_x] == 1:
                            occl_tag = 1
                            break
                elif pix_x < img_cam_pos[0]:
                    k, b = two_points_2_line(point, img_cam_pos)    # 获取两个点所构成的直线，并判断线段中是否有有Occupancy的点
                    # 从两个方向分别进行采样，放置像素化后的稀疏
                    for i_1 in range(pix_x, img_cam_pos[0]):    
                        j_1 = int(np.floor(k * i_1 + b))
                        if scene_occupy[j_1][i_1] == 1:
                            occl_tag = 1
                            break
                    if occl_tag == 0:
                        for j_2 in range(pix_y, img_cam_pos[1]):
                            i_2 = int(np.floor((j_2 - b) / k))
                            if scene_occupy[j_2][i_2] == 1:
                                occl_tag = 1
                                break                      
                elif pix_x > img_cam_pos[0]:
                    k, b = two_points_2_line(point, img_cam_pos)    # 获取两个点所构成的直线，并判断线段中是否有有Occupancy的点
                    # 从两个方向分别进行采样，放置像素化后的稀疏
                    for i_1 in range(img_cam_pos[0] + 1, pix_x + 1):    
                        j_1 = int(np.floor(k * i_1 + b))
                        if scene_occupy[j_1][i_1] == 1:
                            occl_tag = 1
                            break
                    if occl_tag == 0:
                        for j_2 in range(pix_y, img_cam_pos[1]):
                            i_2 = int(np.floor((j_2 - b) / k))
                            if scene_occupy[j_2][i_2] == 1:
                                occl_tag = 1
                                break 
                
                if occl_tag == 1:
                    scene_occlusion[pix_y][pix_x] = 0

        # 画出Occlusion图
        plt.imshow(scene_occlusion, cmap=plt.cm.gray)
        plt.show()

        save_data = {'lidar_id': '%06d'%id, 'occupancy': scene_occupy, 'occlusion': scene_occlusion}
        save.append(save_data)
    
    # with open("D:/1Pjlab/ADModel_Pro/data/GTheatmap_TP_only_car.pkl", "wb") as f:
    #     pickle.dump(save, f)
