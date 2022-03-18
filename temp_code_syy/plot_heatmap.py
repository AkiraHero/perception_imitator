'''
统计最后heatmap的效果
'''

from os import name
import os
import numpy as np
import cv2
import pickle
import math
import matplotlib.pyplot as plt
from PIL import Image
from numpy.testing._private.utils import break_cycles

GTheatmap_file = "D:/1Pjlab/ADModel_Pro/data/GTheatmap.pkl"

with open(GTheatmap_file, 'rb') as f:
    GTheatmap = pickle.load(f)

res = 0.2 # 分辨率0.05m
side_range = (-40, 40)  # 雷达坐标系y轴——左右距离
fwd_range = (0, 70.4)  # 雷达坐标系x轴——后前距离
x_max = 1 + int((side_range[1] - side_range[0]) / res)
y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
all_heatmap = np.zeros([7481, y_max, x_max], dtype=np.uint32)
plot_heatmap = np.zeros([y_max, x_max], dtype=np.uint32)

for i, frm_heatmap in enumerate(GTheatmap):
    all_heatmap[i] = frm_heatmap['heatmap']
    plot_heatmap = plot_heatmap + frm_heatmap['heatmap']

# img_id = 1
# plot_heatmap = all_heatmap[img_id]
plot_heatmap = plot_heatmap/50
plot_heatmap = plot_heatmap.astype(np.uint8)
# plot_heatmap = cv2.applyColorMap(plot_heatmap, cv2.COLORMAP_JET)[...,::-1]
plt.clf()
plt.imshow(plot_heatmap, alpha = 1, cmap="nipy_spectral", vmin=0, vmax=255)
plt.axis('off')
# plt.show()
plt.savefig("./output/plot_heatmap/all_fp_heatmap_2.png", bbox_inches='tight', pad_inches=0.0)
