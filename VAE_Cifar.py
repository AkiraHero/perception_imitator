'''
v2:用于测试VAE_GAN_Mnist_v4
'''

from contextlib import nullcontext
from tarfile import NUL
from matplotlib import image
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

from VAE_GAN_Cifar import VAE

from plot import plot_result_accuracy, plot_result_record

# 加载cifar数据集
def Load_Cifar():
    train_data = torchvision.datasets.CIFAR10( # train_set
        root='./data/',
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        download=True   # 首次使用设为True来下载数据集，之后设为False
    )
    test_data = torchvision.datasets.CIFAR10( # test_set
        root='./data/',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        download=False
    )
    print(f' Size of Train Dataset: {len(train_data)}, Size of Test Dataset: {len(test_data)}')
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    return train_loader, test_loader


# 查看数据（可视化数据）
def datashow(train_loader):
    images, label = next(iter(train_loader))
    images_example = torchvision.utils.make_grid(images)
    images_example = images_example.numpy().transpose(1,2,0) # 将图像的通道值置换到最后的维度，符合图像的格式
    mean = [0.5,0.5,0.5]
    std = [0.5,0.5,0.5]
    images_example = images_example * std + mean
    print(label)
    plt.imshow(images_example )
    plt.show()

# 模型搭建（参考pytorch官网）
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        '''输入为3*32*32，尺寸减半是因为池化层'''
        self.conv1=nn.Sequential(   #建立第一个卷积层
            nn.Conv2d(   #二维卷积层，通过过滤器提取特征
                in_channels=3,   # 图片有几个通道（灰度图片1通道，彩色图片3通道）
                out_channels=16,   # 过滤器也就是卷积核的个数（每一个卷积核都是3通道，和输入图片通道数相同，但是输出的个数依然是卷积核的个数，是因为运算过程中3通道合并为一个通道）
                kernel_size=5,   # 过滤器的宽和高都是5个像素点
                stride=1,   # 每次移动的像素点的个数（步子大小）
                padding=2,   # 在图片周围添加0的层数，stride=1时，padding=(kernel_size-1)/2
            ),   #(3,32,32)-->(16,32,32)
            nn.ReLU(),   #激活函数
            nn.MaxPool2d(kernel_size=2),   # 池化层，压缩特征，一般采用Max方式，kernel_size=2代表在2*2的特征区间内去除最大的) (16,32,32)-->(16,16,16)
        )
        self.conv2=nn.Sequential(   #建立第二个卷积层
            nn.Conv2d(16,32,5,1,2),  # (16,16,16) -->(32,16,16)
            nn.ReLU(),
            nn.MaxPool2d(2),   # (32,16,16)-->(32,8,8)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(32,64,5,1,2),   #(32,8,8)-->(64,8,8)
            nn.ReLU(),
            nn.MaxPool2d(2),   #(64,8,8)-->(64,4,4)
        )
        self.out=nn.Linear(64*4*4,10)   # 全连接层，矩阵相乘形成（1，10）

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=x.view(-1,64*4*4,)   #行数-1代表不知道多少行的情况下，根据原来Tensor内容和Tensor的大小自动分配行数，但是这里为1行
        x=self.out(x)
        return x

# 模型训练
def train(epoch, train_counter, train_losses, train_accs):
    #for epoch in range(EPOCH):  # loop over the dataset multiple times

    #train_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs) # 将数据传入网络进行前向运算
        loss = criterion(outputs, labels) # 得到损失函数
        loss.backward() # 反向传播
        optimizer.step() # 通过梯度做一步参数更新

        # print statistics
        if i % 100 == 99:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (i+1) * len(inputs), len(train_loader.dataset),
                100. * i / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append((i*BATCH_SIZE) + ((epoch-1)*len(train_loader.dataset)))

        correct = 0
        total = 0
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)# labels 的长度
        correct = (predicted == labels).sum().item() # 预测正确的数目
        train_accs.append(100*correct/total)
    print('Finished Training')
    return train_counter, train_losses, train_accs

# 模型测试
def test(net):
    print('\n'+"Begin Testing"+'\n')
    net.eval()
    correct = 0
    total = 0
    test_loss = 0
    accuracy = []
    record = np.zeros((10,10))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            if net == SGAN:
                outputs,_,_ = net(images)   # 返回的一个batch中每张图像对于10个类别的得分
            elif net == Target_model:
                outputs = net(images)
            loss = criterion(outputs, labels) # 得到损失函数
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)   # torch.max(a,1)返回每一行中最大值的那个元素，且返回其索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss /= total
    test_losses.append(test_loss)
    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, total,
    100. * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            if net == SGAN:
                outputs,_,_ = net(images)   # 返回的一个batch中每张图像对于10个类别的得分
            elif net == Target_model:
                outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(100):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                record[labels[i]][predicted[i]] += 1
    for i in range(10):
        print('Accuracy of %s : %2.2f %%' % (i, 100 * class_correct[i] / class_total[i]))
        accuracy.append(class_correct[i] / class_total[i])
    for i in range(10):
        record[i] = record[i] / class_total[i]
        print(record[i])
    print('---------------------------------------------------------------')

    return accuracy, record



if __name__ == '__main__':
    EPOCH = 5
    BATCH_SIZE = 100
    LR = 0.01
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    '''
    transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])
    '''
    train_loader = []
    test_loader = []

    # 加载数据集并显示
    train_loader, test_loader = Load_Cifar()
    # datashow(train_loader)

    train_counter = []
    train_losses = []
    train_accs = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(EPOCH)]

    # 初始化网络结构
    Target_model = Net().to(device=DEVICE)
    SGAN = VAE().to(device=DEVICE)

    # 定义损失函数和优化函数
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=LR)

    # # Train
    # for epoch in range(1, EPOCH + 1):
    #     train_counter, train_losses, train_accs = train(epoch, train_counter, train_losses, train_accs)
    #     torch.save(net.state_dict(), './results/Mnist/param_minist_%d.pt'%(epoch))
    #     test()
    # fig = plt.figure()
    # plt.plot(train_counter, train_losses, color='blue')
    # plt.scatter(test_counter, test_losses, color='red')
    # plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.show()

    # # Test
    Target_model.load_state_dict(torch.load('./results/Cifar/param_minist_10.pt'))
    accuracy_target, record_target = test(Target_model)
    SGAN.load_state_dict(torch.load('./results/VAE_Cifar/model_final.pt'))
    accuracy_SGAN_1, record_SGAN_1 = test(SGAN)
    accuracy_SGAN_2, record_SGAN_2 = test(SGAN)

    plot_result_accuracy(accuracy_target, accuracy_SGAN_1, accuracy_SGAN_2)

    for i in range(10):
        plot_result_record(record_target[i], record_SGAN_1[i], record_SGAN_2[i])
