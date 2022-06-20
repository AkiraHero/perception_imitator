import os
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np

class BaselineDataWrapper:
    def __init__(self, town_name:str, map_root:str):
        self.town_name = town_name
        self.map_root = map_root
        self.world_offset, self.world_scale = self.get_world_offset_and_scale(self.town_name)  # 不同Town对应的Map比例不一致
        self.global_map, self.global_waypoint = self.get_global_map(self.town_name)
        self._geometry = {
            "L1": -40,
            "L2": 40,
            "W1": 0.0,
            "W2": 70.4,
            "label_shape": [352, 400],
            "ratio": 0.2,
        }   # 可以固定下来

    def get_world_offset_and_scale(self, Town):
        offset = {'Town01': [-52.059906005859375, -52.04996085166931],
                'Town02': [-57.45972919464111, 55.3907470703125],
                'Town03': [-199.0638427734375, -259.27125549316406],
                'Town04': [-565.26904296875, -446.1461181640625], 
                'Town05': [-326.0445251464844, -257.8750915527344]
        }
        scale = {'Town01': 5,
                'Town02': 5,
                'Town03': 5,
                'Town04': 5,
                'Town05': 5,
        }
        return offset[Town], scale[Town]
    
    def get_global_map(self, Town):
        map_path = os.path.join(self.map_root, "%s" %Town)
        map = cv2.imread(os.path.join(map_path, "map.png"))
        # 二值化
        map = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
        ret, map = cv2.threshold(map, 0, 1, cv2.THRESH_BINARY)

        for i in range(4):
            if i == 0 :
                waypoint = cv2.imread(os.path.join(map_path, "waypoint%d.png" %i))
                # 二值化
                waypoint = cv2.cvtColor(waypoint, cv2.COLOR_BGR2GRAY)
                ret, waypoint = cv2.threshold(waypoint, 0, 1, cv2.THRESH_BINARY)
            else:
                temp = cv2.imread(os.path.join(map_path, "waypoint%d.png" %i))
                temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
                ret, temp = cv2.threshold(temp, 0, 1, cv2.THRESH_BINARY)
                waypoint = np.clip(waypoint + temp, 0, 1)
        return np.array(map), np.array(waypoint)
    
    def euclidean_distance(self, k ,h , pointIndex):
        '''
        计算一个点到某条直线的euclidean distance
        :param k: 直线的斜率,float类型
        :param h: 直线的截距,float类型
        :param pointIndex: 一个点的坐标,（横坐标,纵坐标）,tuple类型
        :return: 点到直线的euclidean distance,float类型
        '''
        x=pointIndex[0]
        y=pointIndex[1]
        theDistance=math.fabs(h+k*(x-0)-y)/(math.sqrt(k*k+1))
        return theDistance

    def two_points_2_line(self, p1, p2):
        k = (p1[1] - p2[1]) / (p1[0] - p2[0])
        b = p2[1] - k * p2[0]
        return k, b

    def two_points_distance(self, p1, p2):
        dist = math.sqrt(math.pow((p1[1] - p2[1]), 2) +  math.pow((p1[0] - p2[0]), 2))
        return dist

    def get_corners(self, bbox, use_distribution):
        if  use_distribution == True:
            x, y, l, w, yaw, x_var, y_var, l_var, w_var, yaw_var = bbox
            reg_target = [np.cos(yaw), np.sin(yaw), x, y, w, l, yaw_var, x_var, y_var, w_var, l_var]
        else:
            x, y, l, w, yaw = bbox 
            reg_target = [np.cos(yaw), np.sin(yaw), x, y, w, l]
        
        bev_corners = np.zeros((4, 2), dtype=np.float32)
        # rear left
        bev_corners[0, 0] = x - l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[0, 1] = y - l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        # rear right
        bev_corners[1, 0] = x - l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[1, 1] = y - l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front right
        bev_corners[2, 0] = x + l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[2, 1] = y + l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front left
        bev_corners[3, 0] = x + l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[3, 1] = y + l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        return bev_corners, reg_target

    def transform_metric2label(self, metric, ratio=0.2, base_height=352, base_width=200):
        '''
        :param label: numpy array of shape [..., 2] of coordinates in metric space
        :return: numpy array of shape [..., 2] of the same coordinates in label_map space
        '''
        label = - metric / ratio
        label[:, 0] += base_height
        label[:, 1] += base_width

        return label

    def world_to_pixel(self, location, scale, world_offset, offset=(0,0)):
        x = scale * (location[0] - world_offset[0])
        y = scale * (location[1] - world_offset[1])

        return [int(x - offset[0]), int(y - offset[1])]
    
    def get_crop(self, img, ego_pix, meters_behind, meters_ahead, meters_left, meters_right, resolution):
        pix_ahead = meters_ahead * resolution
        pix_behind = meters_behind * resolution
        pix_left = meters_left * resolution
        pix_right = meters_right * resolution

        return img[slice(int(ego_pix[1]-pix_ahead), int(ego_pix[1]-pix_behind)), slice(int(ego_pix[0]+pix_left), int(ego_pix[0]+pix_right)), ...]

    def get_occupancy_and_occlusion(self, kitti_label):
        pic_height, pic_width = self._geometry['label_shape'][:2]
        ratio = self._geometry['ratio']
        img_cam_pos = [pic_height, pic_width/2]
        sample_occupancy = np.zeros([pic_height, pic_width]).astype('int32')
        sample_occlusion = np.ones([pic_height, pic_width]).astype('int32')

        for line in kitti_label:  
            attributes = line.split(" ")   
            if attributes[0] not in ['Car', 'Pedestrian', 'Cyclist', 'TrafficLight', 'TrafficSigns']:
                continue
    
            ############################
            # First stage: get occupancy
            ############################
            cam_x, cam_y, cam_z = attributes[11:14]
            lidar_x = float(cam_z)
            lidar_y = - float(cam_x)
            lidar_l = float(attributes[9])
            lidar_w = float(attributes[10])
            theta = float(attributes[14])

            # 调整坐标到图片坐标系
            img_x, img_y = self.transform_metric2label(np.array([[lidar_x, lidar_y]]))[0].astype('int32')
            img_l = lidar_l / ratio
            img_w = lidar_w / ratio

            # get occupancy
            # 初步筛选,减小计算算量
            pass_size = int(np.ceil(0.5*math.sqrt(img_l**2 + img_w**2)))

            pix_x_min = max(0, img_x - pass_size)
            pix_x_max = min(pic_height, img_x + pass_size)
            pix_y_min = max(0, img_y - pass_size)
            pix_y_max = min(pic_width, img_y + pass_size)

            for pix_x in range(pix_x_min, pix_x_max):
                for pix_y in range(pix_y_min, pix_y_max):
                    w_dis = self.euclidean_distance(np.tan(theta + math.pi), img_y - np.tan(theta + math.pi)*img_x, [pix_x, pix_y])
                    l_dis = self.euclidean_distance(-1/np.tan(theta + math.pi), img_y + 1/np.tan(theta + math.pi)*img_x, [pix_x, pix_y])  

                    if w_dis <= img_w / 2 and l_dis <= img_l / 2:
                        sample_occupancy[pix_x, pix_y] = 1

            ############################                
            # Second stage: get occlusion
            ############################
            corners, _ = self.get_corners([lidar_x, lidar_y, lidar_l, lidar_w, theta], use_distribution=False)
            label_corners = self.transform_metric2label(corners)

            k_list = []
            b_list = [] 
            dist_list = []
            intersection_points = []
            
            if (label_corners < 0).sum() > 0 or (label_corners[:, 0] > 352).sum() == 4 or (label_corners[:, 1] > 352).sum() == 4:
                continue
            
            for i in range(label_corners.shape[0]):
                # 需要截断超出bev视图的部分
                label_corners[i][0] = max(0, label_corners[i][0])
                label_corners[i][0] = min(352, label_corners[i][0])
                label_corners[i][1] = max(0, label_corners[i][1])
                label_corners[i][1] = min(400, label_corners[i][1])

                k, b = self.two_points_2_line(label_corners[i], img_cam_pos)
                dist = self.two_points_distance(label_corners[i], img_cam_pos)

                if k != float("-inf") and  k != float("inf"): 
                    if b < 0:
                        interp_y = 0
                        interp_x = - b / k
                    elif b >= 400:
                        interp_y = 400
                        interp_x = (interp_y - b) / k
                    else:
                        interp_x = 0
                        interp_y = b
                else:
                    if label_corners[i][1] <= 200:
                        b = -1
                        interp_x = 352
                        interp_y = 0
                    else:
                        b =400 + 1
                        interp_x = 352
                        interp_y = 400
                interp = (interp_x, interp_y)

                k_list.append(k)
                b_list.append(b)
                dist_list.append(dist)
                intersection_points.append(interp)

            min_dist_index = dist_list.index(min(dist_list))

            for i in range(0, 3):
                points = []
                if i >= min_dist_index:
                    if (b_list[i+1] < 0 and 0 <= b_list[min_dist_index] < 400) or (b_list[min_dist_index] < 0 and 0 <= b_list[i+1] < 400):
                        points = [label_corners[i+1], label_corners[min_dist_index], intersection_points[min_dist_index], [0,0], intersection_points[i+1]]
                    elif (b_list[i+1] >= 400 and 0 <= b_list[min_dist_index] < 400) or (b_list[min_dist_index] >= 400 and 0 <= b_list[i+1] < 400):
                        points = [label_corners[i+1], label_corners[min_dist_index], intersection_points[min_dist_index], [0,400], intersection_points[i+1]]
                    elif b_list[i+1] >= 400 and b_list[min_dist_index] < 0:
                        points = [label_corners[i+1], label_corners[min_dist_index], intersection_points[min_dist_index], [0,0], [0,400], intersection_points[i+1]]
                    elif b_list[min_dist_index] >= 400 and b_list[i+1] < 0:
                        points = [label_corners[i+1], label_corners[min_dist_index], intersection_points[min_dist_index], [0,400], [0,0], intersection_points[i+1]]
                    else:
                        points = [label_corners[i+1], label_corners[min_dist_index], intersection_points[min_dist_index], intersection_points[i+1]]
                else:
                    if (b_list[i] < 0 and 0 <= b_list[min_dist_index] < 400) or (b_list[min_dist_index] < 0 and 0 <= b_list[i] < 400):
                        points = [label_corners[i], label_corners[min_dist_index], intersection_points[min_dist_index], (0,0), intersection_points[i]]
                    elif (b_list[i] >= 400 and 0 <= b_list[min_dist_index] < 400) or (b_list[min_dist_index] >= 400 and 0 <= b_list[i] < 400):
                        points = [label_corners[i], label_corners[min_dist_index], intersection_points[min_dist_index], [0,400], intersection_points[i]]
                    elif b_list[i] >= 400 and b_list[min_dist_index] < 0:
                        points = [label_corners[i], label_corners[min_dist_index], intersection_points[min_dist_index], [0,0], [0,400], intersection_points[i]]
                    elif b_list[min_dist_index] >= 400 and b_list[i] < 0:
                        points = [label_corners[i], label_corners[min_dist_index], intersection_points[min_dist_index], [0,400], [0,0], intersection_points[i]]
                    else:
                        points = [label_corners[i], label_corners[min_dist_index], intersection_points[min_dist_index], intersection_points[i]]

                points = np.array(points).reshape(-1,1,2).astype(np.int32)
                matrix = np.zeros((pic_width, pic_height), dtype=np.int32)
                cv2.drawContours(matrix, [points], -1, (1), thickness=-1)
                list_of_points_indices = np.nonzero(np.swapaxes(matrix, 1, 0))
                sample_occlusion[list_of_points_indices] = 0
                
        return sample_occupancy, sample_occlusion

    def get_HDmap(self, ego_pose):
        HD_map = []

        ratio = self._geometry['ratio']
        resolution = 1 / ratio
        meters_left = self._geometry['L1']
        meters_right = self._geometry['L2']
        meters_behind = self._geometry['W1']
        meters_ahead = self._geometry['W2']

        position = ego_pose[0:2]
        yaw = ego_pose[4] + 90
        pixel = self.world_to_pixel(position, self.world_scale, self.world_offset)

        for layer in [self.global_map, self.global_waypoint]:
            height, width = layer.shape[:2]
            # 根据egopose计算旋转平移矩阵
            matRotation = cv2.getRotationMatrix2D((pixel), yaw, 1)
            matRotation[0, 2] += width//2 - pixel[0]
            matRotation[1, 2] += height//2 - pixel[1]
            layerRotation = cv2.warpAffine(layer, matRotation,(width,height), borderValue=(0,0,0))
            layerReshape = cv2.resize(layerRotation, (int(width / self.world_scale * resolution), int(height / self.world_scale * resolution)))         # carla map的分辨率不统一,因此需要调整
            layerCrop = self.get_crop(layerReshape, [width / self.world_scale * resolution // 2, height / self.world_scale * resolution // 2], meters_behind, meters_ahead, meters_left, meters_right, resolution)
            HD_map.append(layerCrop)

        HD_map = np.stack(HD_map, axis=0)

        return HD_map

    def get_frame_data(self, ego_pose, kitti_label):
        occupancy, occlusion = data_wrapper.get_occupancy_and_occlusion(kitti_label)
        map = data_wrapper.get_HDmap(ego_pose)

        model_input = np.concatenate((occupancy[None, ...], occlusion[None, ...], map), axis=0)

        data_dict = {
            'occupancy' : occupancy,
            'occlusion' : occlusion,
            'map' : map,
            'input' : model_input
        }

        plt.imshow(occupancy)
        plt.show()
        plt.imshow(map[1])
        plt.show()

        return data_dict
    

    
if __name__ == '__main__':
    # carla data for test
    Town_name = 'Town05'
    map_root = "E:/PJLAB_Experiment/Data/carla/Maps"
    kitti_label = open("E:/PJLAB_Experiment/Data/carla/carla_new/label_2/011537.txt").readlines()
    ego_pose = np.load("E:/PJLAB_Experiment/Data/carla/carla_new/pose/011537.npy")
    # ego_pose = [5.141849994659424, -201.4500732421875, 0.0629693940281868, 0.3861323595046997, 1.233340859413147, -159.46835327148438]
    data_wrapper = BaselineDataWrapper(Town_name, map_root)
    data = data_wrapper.get_frame_data(ego_pose, kitti_label)
    print(data['input'].shape)




