import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader
from tensorboardX.writer import SummaryWriter


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(8, 100)
        self.fc2 = nn.Linear(100, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        x3 = torch.sigmoid(self.fc3(x2))
        return x3

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


class SimpleDataset(Dataset):
    def __init__(self, is_train=True):
        super(Dataset, self).__init__()
        self.is_train = is_train
        db_file = "/home/akira/Downloads/kitti_pvrcnn_all.pkl"
        with open(db_file, 'rb') as f:
            self.db = pickle.load(f)
        self.train_frm_ratio = 0.6
        train_test_dataset_division_file = "train_test_sample.pkl"
        train_test_dataset_division = None
        if os.path.exists(train_test_dataset_division_file):
            with open(train_test_dataset_division_file, 'rb') as f:
                train_test_dataset_division = pickle.load(f)
                train_list = train_test_dataset_division['train_list']
                test_list = train_test_dataset_division['test_list']

        else:
            train_list, test_list = self.sample_dataset()
            d = {
                'train_list': train_list,
                'test_list': test_list
            }
            with open(train_test_dataset_division_file, 'wb') as f:
                pickle.dump(d, f)
        self.train_db = [self.db[i] for i in train_list]
        self.test_db = [self.db[i] for i in test_list]
        if is_train:
            self.item_list = get_data_dict(self.train_db)
            pass
        else:
            self.item_list = get_data_dict(self.test_db)

    def sample_dataset(self):
        frm_num = len(self.db)
        train_list = []
        test_list = []
        for i in range(frm_num):
            if np.random.rand() < self.train_frm_ratio:
                train_list.append(i)
            else:
                test_list.append(i)
        return train_list, test_list

    def __getitem__(self, index):
        return self.item_list[index]

    def __len__(self):
        return len(self.item_list)

    @staticmethod
    def batch_collate_fn(data):
        batch_size = len(data)
        keys = data[0].keys()
        batch_data_dict = {
            key: np.array([i[key] for i in data]).reshape(batch_size, -1) for key in keys
        }
        return batch_data_dict

    def get_data_loader(self):
        data_loader = DataLoader(
            dataset=self,
            batch_size=1024,
            shuffle=True if self.is_train else False,
            num_workers=0,
            collate_fn=self.batch_collate_fn
        )
        return data_loader

def load_data2gpu(data):
    for k, v in data.items():
        v = torch.from_numpy(v).to(torch.float)
        v = v.cuda()
        data[k] = v

# train a mlp to infer detected or not:
# method1: input box itself
#######################################################################################
# dataset = SimpleDataset()
# data_loader = dataset.get_data_loader()
# max_epoch = 800
# model = SimpleMLP()
# model.cuda()
# optimizer = torch.optim.RMSprop(lr=0.001, params=model.parameters())
# loss_func = nn.BCELoss(reduce=False)
# torch.autograd.set_detect_anomaly(True)
# writer = SummaryWriter(logdir="/home/akira/tmp_tb")
# iter = 0
# for epoch in range(max_epoch):
#     for step, data in enumerate(data_loader):
#         optimizer.zero_grad()
#         model.zero_grad()
#         load_data2gpu(data)
#         input_ = data['gt_box']
#         batch_size = input_.shape[0]
#         target = data['detected']
#         loss_weight = torch.ones(target.shape, device=target.device)
#         detected_inx = (target == 1).nonzero()
#         non_detected_inx = (target == 0).nonzero()
#
#         loss_weight[detected_inx] = 0.18084
#         loss_weight[non_detected_inx] = 0.81916
#
#         output = model(input_)
#         loss = loss_func(output, target).mul(loss_weight).sum() / batch_size
#
#         loss.backward()
#         optimizer.step()
#         print("epoch{}, step{}, Loss={}".format(epoch, step, loss))
#         iter += 1
#         writer.add_scalar("loss", loss.item(), global_step=iter)
#     if epoch % 50 == 0:
#         torch.save(model.state_dict(), "model_detected-epoch{}.pt".format(epoch))
# torch.save(model.state_dict(), "model_detected-epoch{}.pt".format(epoch))
###############################################################################

# test
dataset_test = SimpleDataset(is_train=False)
data_loader_test = dataset_test.get_data_loader()
target_epoch = 300
para_file = "model_detected-epoch{}.pt".format(target_epoch)
model = SimpleMLP()
model.cuda()
model.load_state_dict(torch.load(para_file))
model.eval()
score_thres = 0.5

tt_detected = 0
tt_not_detected = 0
tt_detected_dt_detected = 0
tt_detected_dt_not_detected = 0
tt_not_detected_dt_detected = 0
tt_not_detected_dt_not_detected = 0
for step, data in enumerate(data_loader_test):
    load_data2gpu(data)
    input_ = data['gt_box']
    batch_size = input_.shape[0]
    target = data['detected']
    output = model(input_)
    output_cls = torch.zeros(output.shape, device=output.device)
    detected_inx = (output > score_thres).nonzero()
    output_cls[detected_inx] = 1
    tt_detected += (target == 1).nonzero().sum()
    tt_not_detected += (target == 0).nonzero().sum()
    tt_detected_dt_detected += ((target == 1) & (output_cls == 1)).nonzero().sum()
    tt_detected_dt_not_detected += ((target == 1) & (output_cls == 0)).nonzero().sum()
    tt_not_detected_dt_detected += ((target == 0) & (output_cls == 1)).nonzero().sum()
    tt_not_detected_dt_not_detected += ((target == 0) & (output_cls == 0)).nonzero().sum()

print("========== evaluation, score_thres={}, epoch={}============".format(score_thres, target_epoch))
print("TP:detected", tt_detected_dt_detected / tt_detected)
print("TP:non_detected", tt_not_detected_dt_not_detected / tt_not_detected)
print("FN:detected / FP:non_detected", tt_detected_dt_not_detected / tt_detected)
print("FN:non_detected / FP: detected", tt_not_detected_dt_detected / tt_not_detected)
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