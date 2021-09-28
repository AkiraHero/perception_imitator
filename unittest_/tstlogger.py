



import torch
from torch import nn
import torch.nn.functional as func

import functools
import operator

def shape_of_output(shape_of_input, list_of_layers):
    sequential = nn.Sequential(*list_of_layers)
    return tuple(sequential(torch.rand(1, *shape_of_input)).shape)

def size_of_output(shape_of_input, list_of_layers):
    return functools.reduce(operator.mul, list(shape_of_output(shape_of_input, list_of_layers)))


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3, 1, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)






    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = func.max_pool2d(func.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = func.max_pool2d(func.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)     # x为10个类别的得分
        # x = F.sigmoid(x)
        return x
a = nn.Conv2d(1, 6, 3, 1, 2)
pass