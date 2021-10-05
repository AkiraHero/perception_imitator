import torch
import torch.nn as nn
from tensorboardX.writer import SummaryWriter
from dataset.box_only_dataset import BoxOnlyDataset


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(8, 100)
        self.fc2 = nn.Linear(100, 64)
        self.fc_x = nn.Linear(64, 6)
        self.fc_y = nn.Linear(64, 6)
        self.fc_z = nn.Linear(64, 6)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x_f = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc_x(x_f))
        y = torch.sigmoid(self.fc_y(x_f))
        z = torch.sigmoid(self.fc_z(x_f))
        return x, y, z

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
    config = {
        'paras': {
            "for_train": True,
            "batch_size": 2048,
            "data_root": "../../data/kitti",
            "num_workers": 0,
            "load_all": False,
            "screen_no_dt": True
        }
    }
    dataset = BoxOnlyDataset(config)
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
            BoxOnlyDataset.load_data2gpu(data)
            input_ = data['gt_box']
            batch_size = input_.shape[0]
            target_x = data['box_diff_cls'][:, 0, :].long()
            target_y = data['box_diff_cls'][:, 1, :].long()
            target_z = data['box_diff_cls'][:, 2, :].long()


            output_mu = model(input_)
            loss = loss_func(output_mu[0], target_x.reshape(-1,)) \
                   + loss_func(output_mu[1], target_y.reshape(-1,)) \
                    + loss_func(output_mu[2], target_z.reshape(-1, ))

            loss.backward()
            optimizer.step()
            print("epoch{}, step{}, Loss={}".format(epoch, step, loss))
            iter += 1
            writer.add_scalar("loss_loc_discrete_x", loss.item(), global_step=iter)
        if epoch % 100 == 0:
            torch.save(model.state_dict(), "model_loc_discrete_x-epoch{}.pt".format(epoch))
    torch.save(model.state_dict(), "model_loc_discrete_x-epoch{}.pt".format(epoch))
    ##############################################################################
else:
    # test
    with torch.no_grad():
        config = {
            'paras': {
                "for_train": False,
                "batch_size": 100000000000000,
                "data_root": "../../data/kitti",
                "num_workers": 0,
                "load_all": False,
                "screen_no_dt": True
            }
        }
        dataset_test = BoxOnlyDataset(config)
        data_loader_test = dataset_test.get_data_loader()
        target_epoch = 900
        para_file = "model_loc_discrete_x-epoch{}.pt".format(target_epoch)
        model = SimpleMLP()
        model.cuda()
        model.load_state_dict(torch.load(para_file))
        model.eval()
        score_thres = 0.5


        for step, data in enumerate(data_loader_test):

            BoxOnlyDataset.load_data2gpu(data)
            input_ = data['gt_box']
            batch_size = input_.shape[0]

            target_x = data['box_diff_cls'][:, 0, :].squeeze(1)
            target_y = data['box_diff_cls'][:, 1, :].squeeze(1)
            target_z = data['box_diff_cls'][:, 2, :].squeeze(1)
            output_x, output_y, output_z = model(input_)
            _, cls_pred_x= output_x.max(dim=1)
            _, cls_pred_y= output_y.max(dim=1)
            _, cls_pred_z= output_z.max(dim=1)

            tp_inx_x = (target_x == cls_pred_x).nonzero()
            tp_inx_y = (target_y == cls_pred_y).nonzero()
            tp_inx_z = (target_z == cls_pred_z).nonzero()

            print("========== evaluation, epoch={}============".format(target_epoch))
            print("x_acc", len(tp_inx_x)/len(cls_pred_x))
            print("y_acc", len(tp_inx_y)/len(cls_pred_y))
            print("z_acc", len(tp_inx_z)/len(cls_pred_z))



            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            label_x = [ "({:.2f}~{:.2f})".format(i[0], i[1]) for i in dataset_test.segs[0]]
            label_y = [ "({:.2f}~{:.2f})".format(i[0], i[1]) for i in dataset_test.segs[1]]
            label_z = [ "({:.2f}~{:.2f})".format(i[0], i[1]) for i in dataset_test.segs[2]]

            C = confusion_matrix(target_x.cpu(), cls_pred_x.cpu(), normalize='true')
            disp = ConfusionMatrixDisplay(confusion_matrix=C)
            disp.plot()
            # plt.show()
            plt.savefig("x_confuse.png")

            C = confusion_matrix(target_y.cpu(), cls_pred_y.cpu(), normalize='true')
            disp = ConfusionMatrixDisplay(confusion_matrix=C)
            disp.plot()
            # plt.show()
            plt.savefig("y_confuse.png")

            C = confusion_matrix(target_z.cpu(), cls_pred_z.cpu(), normalize='true')
            disp = ConfusionMatrixDisplay(confusion_matrix=C)
            disp.plot()
            # plt.show()
            plt.savefig("z_confuse.png")
            pass



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