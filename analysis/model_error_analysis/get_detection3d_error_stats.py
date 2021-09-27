import matplotlib.pyplot as plt
import numpy as np
from .plot_binbox import plot_binbox

# dataset self

# error class:
    # total
        # detected/TP FN
        # location x/y/z error
        # dimension x/y/z error
        # rotation error
    # marginalizing by class
    # marginalizing by every dim

'''
box:
    N * [location(3)+dimension(3)+rotation(1)+class(1)]
marginalize_list=[
    {
        dim = 0,
        segments/value = []    
    }

]
'''

def get_error(box_gt, box_dt):
    # get box diff
    box_diff = box_dt[:, :7] - box_gt[:, :7]
    rot_diff = np.arctan(np.tan(box_diff[:, 6]))
    box_diff[:, 6] = rot_diff
    return box_diff


def marginalize(data, margin, condition_list, margin_by="value"):
    if margin_by == "segment":
        assert isinstance(condition_list[0], tuple)
    indices_list = []
    assert len(data) ==  len(margin)
    if margin_by == "value":
        for i in condition_list:
            indices = (margin == i).nonzero()
            indices_list.append(indices)
    if margin_by == 'segment':
        for i in condition_list:
            indices = (i[0] <= margin & margin < i[1]).nonzero()
            indices_list.append(indices)
    return indices_list


def plot_error_statistics(box_gt, box_dt):
    box_diff = get_error(box_gt, box_dt)
    datadict = {}
    datadict['x'] = {'data': box_diff[:, 0], 'label': 'x'}
    datadict['y'] = {'data': box_diff[:, 1], 'label': 'y'}
    datadict['z'] = {'data': box_diff[:, 2], 'label': 'z'}
    datadict['w'] = {'data': box_diff[:, 3], 'label': 'w'}
    datadict['h'] = {'data': box_diff[:, 4], 'label': 'h'}
    datadict['l'] = {'data': box_diff[:, 5], 'label': 'l'}
    datadict['r'] = {'data': box_diff[:, 6], 'label': 'rot'}
    fig1 = plot_binbox(datadict, "Box Error Distribution", size=(5, 5))
    return fig1


def get_error_segs(box_gt, box_dt, class_num=4):
    box_diff = get_error(box_gt, box_dt)
    seg_list = []
    for i in range(7):
        sub_seg_list = []
        sorted_vec = np.sort(box_diff[:, i])
        length = len(sorted_vec)
        step = length // class_num
        now = 0
        for j in range(class_num):
            ed = now + step
            if ed >= length:
                ed = length - 1
            sub_seg_list.append([sorted_vec[now], sorted_vec[ed]])
            now = now + step
        seg_list.append(sub_seg_list)
    return seg_list

def get_discrete_distribution_diff(sq_target, sq_model, seg_list):
    seg_target_list = []
    seg_predict_list = []
    seg_intersection_list = []
    for i in seg_list:
        model_inx, = ((sq_model >= i[0]) & (sq_model < i[1])).nonzero()
        target_inx, = ((sq_target >= i[0]) & (sq_target < i[1])).nonzero()
        seg_target_list.append(len(target_inx))
        seg_predict_list.append(len(model_inx))
        seg_intersection_list.append(len(np.intersect1d(target_inx, model_inx)))
    return seg_target_list, seg_predict_list, seg_intersection_list


