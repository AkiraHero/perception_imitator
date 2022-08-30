import torch
import torch.nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import os

def get_bev(velo_array, label_list = None, scores = None):
    map_height, map_width = velo_array.shape[0:2]
    intensity = np.zeros((velo_array.shape[0], velo_array.shape[1], 3), dtype=np.uint8)   
    val = velo_array[..., -1] * 255
    # val = (1 - velo_array[::-1, :, :-1].max(axis=2)) * 255
    intensity[:, :, 0] = val
    intensity[:, :, 1] = val
    intensity[:, :, 2] = val

    if label_list is not None:
        for corners in label_list:
            plot_corners = - corners / 0.2
            plot_corners[:, 0] += int(map_height )
            plot_corners[:, 1] += int(map_width // 2)
            # 交换坐标xy
            plot_corners = plot_corners[:, ::-1]

            plot_corners = plot_corners.astype(int).reshape((-1, 1, 2))
            cv2.polylines(intensity, [plot_corners], True, (255, 0, 0), 2)
            cv2.line(intensity, tuple(plot_corners[2, 0]), tuple(plot_corners[3, 0]), (0, 0, 255), 3)

    return intensity

def plot_bev(velo_array, label_list = None, scores = None, window_name='GT', save_path=None):
    '''
    Plot a Birds Eye View Lidar and Bounding boxes (Using OpenCV!)
    The heading of the vehicle is marked as a red line
        (which connects front right and front left corner)

    :param velo_array: a 2d velodyne points
    :param label_list: a list of numpy arrays of shape [4, 2], which corresponds to the 4 corners' (x, y)
    The corners should be in the following sequence:
    rear left, rear right, front right and front left
    :param window_name: name of the open_cv2 window
    :return: None
    '''

    intensity = get_bev(velo_array, label_list, scores)

    if save_path != None:
        print(save_path)
        cv2.imwrite(save_path, intensity)
        cv2.waitKey(0)
    else:
        # cv2.imshow(window_name, intensity)
        # cv2.waitKey(0)
        pass

    return intensity

def plot_label_map(label_map):
    plt.figure()
    plt.imshow(label_map)
    plt.show()