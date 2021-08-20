import torch.nn as nn
import torch.nn.functional as F

class Discriminator_MLP(nn.Module):
    def __init__(self, ngpu, input_size):
        super(Discriminator_MLP, self).__init__()
        self.ngpu = ngpu
        # 初始化四层神经网络 两个全连接的隐藏层，一个输出层
        self.fc1 = nn.Linear(input_size,200) # 第一个隐含层
        self.fc2 = nn.Linear(200,100) # 第二个隐含层
        self.fc3 = nn.Linear(100,1)  # 输出层
        self.dropout = nn.Dropout(p=0.5)    # Dropout暂时先不打开

    def forward(self, din):
        # 前向传播， 输入值：din, 返回值 dout
        din = din.view(-1,din.shape[2])    # 将一个多行的Tensor,拼接成一行
        dout = self.fc1(din)
        dout = F.relu(dout)  # 使用 relu 激活函数
        dout = self.fc2(dout)
        dout = F.relu(dout)
        dout = self.fc3(dout)
        dout = F.sigmoid(dout)
        # dout = F.softmax(dout, dim=1) # 输出层使用 softmax 激活函数,这里输出层为1,因此不需要softmax,使用sigmoid
        return dout