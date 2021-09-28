import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader
from tensorboardX.writer import SummaryWriter
from analysis.model_error_analysis.get_detection3d_error_stats import get_error, plot_error_statistics, get_discrete_distribution_diff
from boxonly_dataset import SimpleDataset, load_data2gpu

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(8, 100)
        self.fc2 = nn.Linear(100, 64)
        self.fc_mu = nn.Linear(64, 1)
        self.fc_logvar = nn.Linear(64, 1)


    def forward(self, x):
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        mu = self.fc_mu(x2)
        log_var = self.fc_logvar(x2)
        return mu, log_var

def label_str2num(label):
    d = {
        'Car': 1,
        'Pedestrian': 2,
        'Cyclist': 3
    }
    return d[label]


def get_data_dict(db):
    # sum up all frm
    datadict_list = []
    for frm in db:
        for inx, matched_obj_inx in enumerate(frm['gt_valid_inx']):
            gt = frm['ini_frm_annos']['gt_boxes_lidar']
            cur_gt_box = gt[matched_obj_inx]
            cur_dt_box = frm['ordered_lidar_boxes'][inx]
            cur_gt_label = label_str2num(frm['ini_frm_annos']['name'][matched_obj_inx])
            if np.sum(cur_dt_box).astype(np.int) == 0:
                detected = 0
            else:
                detected = 1
            box_diff = cur_dt_box[:7] - cur_gt_box
            rot_diff = np.arctan(np.tan(box_diff[6]))
            box_diff[6] = rot_diff
            cur_gt_box[6] = np.arctan(np.tan(cur_gt_box[6]))
            cur_dt_box[6] = np.arctan(np.tan(cur_dt_box[6]))
            data_dict = {'detected': detected, 'box_diff': box_diff,
                         'gt_box': np.concatenate([cur_gt_box, np.array([cur_gt_label])]), 'dt_box': cur_dt_box}
            datadict_list.append(data_dict)
    return datadict_list




is_train = 1
# train a mlp to infer detected or not:
# method1: input box itself
#######################################################################################
if is_train:
    dataset = SimpleDataset(screen_no_dt=True)
    data_loader = dataset.get_data_loader()
    max_epoch = 1800
    model = SimpleMLP()
    model.cuda()
    optimizer = torch.optim.RMSprop(lr=0.001, params=model.parameters())
    loss_func = nn.GaussianNLLLoss()
    torch.autograd.set_detect_anomaly(True)
    writer = SummaryWriter(logdir="/home/akira/tmp_tb")
    iter = 0
    for epoch in range(max_epoch):
        for step, data in enumerate(data_loader):
            optimizer.zero_grad()
            model.zero_grad()
            load_data2gpu(data)
            input_ = data['gt_box']
            batch_size = input_.shape[0]
            target = data['box_diff'][:, 6]
            # loss_weight = torch.ones(target.shape, device=target.device)
            # detected_inx = (target == 1).nonzero()
            # non_detected_inx = (target == 0).nonzero()
            #
            # loss_weight[detected_inx] = 0.18084
            # loss_weight[non_detected_inx] = 0.81916

            output_mu, output_logvar = model(input_)
            loss = loss_func(output_mu, target, output_logvar.exp())

            loss.backward()
            optimizer.step()
            print("epoch{}, step{}, Loss={}".format(epoch, step, loss))
            iter += 1
            writer.add_scalar("loss_rot", loss.item(), global_step=iter)
        if epoch % 50 == 0:
            torch.save(model.state_dict(), "model_rot-epoch{}.pt".format(epoch))
    torch.save(model.state_dict(), "model_rot-epoch{}.pt".format(epoch))
    ##############################################################################
else:
    # test
    with torch.no_grad():
        dataset_test = SimpleDataset(is_train=False, batch_size=100000000000000, screen_no_dt=True)
        data_loader_test = dataset_test.get_data_loader()
        target_epoch = 1799
        para_file = "model_rot-epoch{}.pt".format(target_epoch)
        model = SimpleMLP()
        model.cuda()
        model.load_state_dict(torch.load(para_file))
        model.eval()
        score_thres = 0.5

        targets = []
        model_mu = []
        model_logvar = []
        for step, data in enumerate(data_loader_test):
            load_data2gpu(data)
            input_ = data['gt_box']
            batch_size = input_.shape[0]
            target = data['box_diff'][:, 6]
            output_mu, output_logvar = model(input_)
            targets.append(target.cpu().numpy())
            model_mu.append(output_mu.cpu().numpy())
            model_logvar.append(output_logvar.cpu().numpy())
        pass
        targets = np.concatenate(targets, axis=0)
        model_mu = np.concatenate(model_mu, axis=0)
        model_logvar = np.concatenate(model_logvar, axis=0)

        print("========== evaluation, epoch={}============".format(target_epoch))
        eval_seg = [
            [-10, -1.0],
            [-1.0, -0.5],
            [-0.5, 0.0],
            [0.0, 0.5],
            [0.5, 1.0],
            [1.0, 10]
        ]
        tar_sum_x, pre_sum_x, inter_sum_x = get_discrete_distribution_diff(targets[:, 0], model_mu[:, 0], eval_seg)
        print("========== evaluation, epoch={}============".format(target_epoch))
        print("rot_acc", sum(inter_sum_x) / sum(tar_sum_x))

# method1: input frm boxes

# get loc error distribution when detected: stats: cov
# assume gaussian?
# try to fit it use mlp:
    # method 1: use box itself
    # method 2: use box all

# get dim error using same trick

# get class prediction use same trick

# get rot error use same trick

# thinking: how to use all boxes: graph network.


pass