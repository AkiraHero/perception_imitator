import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader
from tensorboardX.writer import SummaryWriter
from analysis.model_error_analysis.get_detection3d_error_stats import get_error, plot_error_statistics, get_discrete_distribution_diff, get_error_segs
from boxonly_dataset import SimpleDataset, load_data2gpu


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(8, 100)
        self.fc2 = nn.Linear(100, 64)
        self.fc_x = nn.Linear(64, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc_x(x))
        return x

def label_str2num(label):
    d = {
        'Car': 1,
        'Pedestrian': 2,
        'Cyclist': 3
    }
    return d[label]





is_train = 0
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
    loss_func = nn.CrossEntropyLoss()
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
            target = data['box_diff_cls'][:, 0, :].long()


            output_mu = model(input_)
            loss = loss_func(output_mu, target.reshape(-1,))

            loss.backward()
            optimizer.step()
            print("epoch{}, step{}, Loss={}".format(epoch, step, loss))
            iter += 1
            writer.add_scalar("loss_loc_discrete_x", loss.item(), global_step=iter)
        if epoch % 50 == 0:
            torch.save(model.state_dict(), "model_loc_discrete_x-epoch{}.pt".format(epoch))
    torch.save(model.state_dict(), "model_loc_discrete_x-epoch{}.pt".format(epoch))
    ##############################################################################
else:
    # test
    with torch.no_grad():
        dataset_test = SimpleDataset(is_train=False, batch_size=100000000000000, screen_no_dt=True)
        data_loader_test = dataset_test.get_data_loader()
        target_epoch = 1000
        para_file = "model_loc_discrete_x-epoch{}.pt".format(target_epoch)
        model = SimpleMLP()
        model.cuda()
        model.load_state_dict(torch.load(para_file))
        model.eval()
        score_thres = 0.5


        for step, data in enumerate(data_loader_test):

            load_data2gpu(data)
            input_ = data['gt_box']
            batch_size = input_.shape[0]
            target = data['box_diff_cls'][:, 0, :].squeeze(1)
            output_ = model(input_)
            _, cls_pred= output_.max(dim=1)
            tp_inx = (target == cls_pred).nonzero()

            print("========== evaluation, epoch={}============".format(target_epoch))
            print("x_acc", len(tp_inx)/len(cls_pred))

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