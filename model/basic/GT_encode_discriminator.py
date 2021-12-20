from model.model_base import ModelBase
import factory.model_factory as ModelFactory
import torch.nn as nn
import torch.nn.functional as F
import torch

'''
multiple mlp for discriminator
    inpupts: 640 GT + 140 GenFP
'''

class GTEncodeDiscriminator(ModelBase):
    def __init__(self, config):
        super(GTEncodeDiscriminator, self).__init__()
        self.fc_GT = nn.Linear(8, 1)
        self.fc_FP = nn.Linear(7, 1)
        self.fc_all = nn.Linear(100, 2048)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x):   # x:[bs, 640+140]
        x_GT = x[:, 0:640]
        x_GT = x_GT.reshape(-1,8)  # x_GT:[bs*80, 8]
        x_FP = x[:, 640:]     # x_FP:[bs*140]
        x_FP = x_FP.reshape(-1,7)  # x_FP:[bs*20, 7]

        x_GT = F.relu(self.fc_GT(x_GT)).reshape(-1,80)
        x_FP = F.relu(self.fc_FP(x_FP)).reshape(-1,20)
        x = torch.cat((x_GT, x_FP), 1)

        x = F.relu(self.fc_all(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(F.relu(self.fc3(x)))
        return x