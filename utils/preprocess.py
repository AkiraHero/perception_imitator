import math
import os
import pickle
import numpy as np

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

def two_points_distance(p1, p2):
    dist = math.sqrt(math.pow((p1[1] - p2[1]), 2) +  math.pow((p1[0] - p2[0]), 2))
    return dist
