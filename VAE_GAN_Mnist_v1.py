'''
在原先GAN的基础上引入VAE的思想，此代码尝试VAE：从 源图像 编码-> 隐向量 -> 图片+10个类别得分
'''
from pickle import DICT
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os

from torch.cuda import device_of
from torch._C import device
from torch.types import Device
from torch.utils import data
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, dataloader, sampler
from torchvision import utils, datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from IPython.display import HTML
import numpy as np

from CNN_Mnist import Net


# Set random seed for reproducibility
torch.manual_seed(0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# 定义画图工具
def show_images(images, name): 
    plt.rcParams['figure.figsize'] = (10.0, 8.0) # 设置画图的尺寸
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    images = np.reshape(images, [images.shape[0], -1])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    plt.savefig(name)
    return
def deprocess_img(x):   # 将(-1,1)变换到(0,1)
    return (x + 1.0) / 2.0    

def Combine_data(data, label):     # 直接对tensor类型进行处理，这样可以保存反传的梯度，将处理后图片与经过G得到的类别组合成可以输入D的数据
    pic_len = data.shape[1]*data.shape[2]*data.shape[3]    # 获取每一张图片压缩后的总像素个数
        
    img_Din = data.squeeze().reshape((data.shape[0],1,pic_len))   # 变为batch_size*1*len
    # label_Din = label.cpu().unsqueeze(-1).unsqueeze(-1).numpy()   # 获得对应label
    label_Din = label.unsqueeze(-2)   # 获得对应label,对于增添10各类别概率仅需要加一个维度即可
    img_Din = torch.cat((img_Din, label_Din), 2)    # 将label余图像直接tensor合并，得到batch_size*1*(len+10)，主要为了是的tensor能够用保留反传梯度
    # img_Din = torch.from_numpy(np.append(img_Din, label_Din, axis=2)) # 将label加入得到batch_size*1*(len+10)
    # img_Din = img_Din.to(torch.float32) # 将double精度转为float适用于全连接层输入类型

    return img_Din

def Load_Mnist_ori():
    train_data = datasets.MNIST( # train_set
        root=dataroot,
        train=True,
        transform=transforms.Compose([
            # transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        download=True   # 首次使用设为True来下载数据集，之后设为False
    )
    dataset = train_data
    # test_data = datasets.MNIST( # test_set
    #     root=dataroot,
    #     train=False,
    #     transform=transforms.ToTensor(),
    #     download=True
    # )
    # dataset = train_data+test_data
    print(f'Total Size of Dataset: {len(dataset)}')

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers
    )
    return dataloader

# 定义VAE中编码器，将数据进行编码并重参化为mean和var
class Encoder(nn.Module):
    def __init__(self, ngpu):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20) # 得到mean
        self.fc22 = nn.Linear(400, 20) # 得到var

    def encode(self, x): # 编码层
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):    #最后得到的是u(x)+sigma(x)*N(0,I)
        std = logvar.mul(0.5).exp_() #e**(x/2)
        eps = torch.FloatTensor(std.size()).normal_()   # 用正态分布填充eps
        if torch.cuda.is_available():
            eps = Variable(eps.cuda())
        else:
            eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x) # 编码
        z = self.reparametrize(mu, logvar) # 重新参数化成正态分布
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        return z,KLD # 解码，同时输出均值方差 

# VAE的解码器，同时也是GAN的生成器，此处让其直接生成是图片+得分(28*28+10)
class Decoder(nn.Module): 
    def __init__(self, ngpu, noise_dim=20):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 500),
            nn.ReLU(True),
            nn.BatchNorm1d(500),
            nn.Linear(500, 128*7*7),
            nn.ReLU(True),
            nn.BatchNorm1d(128*7*7)
        )
        
        # 用于图像的生成，bs*128*7*7 ----> ns*1*28*28
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),   
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, 2, padding=1),#128,1,1,1
            nn.Tanh()
        )

        # 用于图像的分类，得到十类得分
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3, 1, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        # 生成的新图
        dout1 = self.fc(x)  
        dout1 = dout1.view(dout1.shape[0], 128, 7, 7) # reshape通道是 128，大小是 7x7
        dout1 = self.conv(dout1)
        # 分类结果
        dout2 = F.max_pool2d(F.relu(self.conv1(dout1)), (2, 2))
        dout2 = F.max_pool2d(F.relu(self.conv2(dout2)), 2)
        dout2 = dout2.view(-1, self.num_flat_features(dout2))
        dout2 = F.relu(self.fc1(dout2))
        dout2 = F.relu(self.fc2(dout2))
        dout2 = self.fc3(dout2)     # x为10个类别的得分
        # 将图片和类别合并
        dout1 = dout1.reshape((dout1.shape[0], dout1.shape[1]*dout1.shape[2]*dout1.shape[3]))
        y = torch.cat((dout1, dout2), 1).unsqueeze(1)
        y = Variable(y)

        return y

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Discriminator_MLP(nn.Module):
    def __init__(self, ngpu, input_size):
        super(Discriminator_MLP, self).__init__()
        self.ngpu = ngpu
        # 初始化四层神经网络 两个全连接的隐藏层，一个输出层
        self.fc1 = nn.Linear(input_size,2048) # 第一个隐含层
        self.fc2 = nn.Linear(2048,2048) # 第二个隐含层
        self.fc3 = nn.Linear(2048,1)  # 输出层
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

def lossD(scores_real, scores_fake0, scores_fake1):
    loss = 0.5 * ((scores_real - 1) ** 2).mean() + 0.5 * (scores_fake0 ** 2).mean()+0.5 * (scores_fake1 ** 2).mean()
    return loss

def lossGD(scores_fake):
    loss = 0.5 * ((scores_fake - 1) ** 2).mean()
    return loss


if __name__ == '__main__':
    dataroot = "/home/PJLAB/sunyiyang/桌面/PJlab/GAN_Exp/Datasets"  # Root directory for dataset
    workers = 10    # Number of workers for dataloader
    batch_size = 100     # Batch size during training
    ngpu = 1        # Number of GPUs available. Use 0 for CPU mode.
    num_epochs = 100
    noise_dim = 20    # 隐向量及输入随机生成的噪声向量维度
    input_size = 28*28 + 10

    DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')

    dataloaders = Load_Mnist_ori()
    
    # 加载Target CNN模型，用于得到参照的真实分类
    Tar_CNN= Net().to(device=DEVICE)
    Tar_CNN.load_state_dict(torch.load('./results/Mnist/param_minist_5.pt'))

    # 建立E、G和D
    Enc = Encoder(ngpu).to(device=DEVICE)
    Enc.apply(weights_init)
    Gen = Decoder(ngpu, noise_dim).to(device=DEVICE)
    Gen.apply(weights_init)
    Dis = Discriminator_MLP(ngpu, input_size).to(device=DEVICE)
    Dis.apply(weights_init)

    # 初始化Optimizers
    E_trainer = torch.optim.Adam(Enc.parameters(), lr=1e-3)
    G_trainer = torch.optim.Adam(Gen.parameters(), lr=3e-4, betas=(0.5, 0.999))
    D_trainer = torch.optim.Adam(Dis.parameters(), lr=3e-4, betas=(0.5, 0.999))

    # 开始训练
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        beg_time = time.time()
        for i, data in enumerate(dataloaders):
            # 先将数据集通过TargetCNN网络得到类别，并和原图合并作为真值
            real_input_pic = data[0].to(device=DEVICE)  # 原图片bs*1*28*28
            real_input_pic0 = real_input_pic.view(batch_size, -1)   # 将原图摊平bs*784
            Target_score = Tar_CNN(real_input_pic)
            real_input = Combine_data(real_input_pic, Target_score) # 得到bs*1*(784+10)真值
            # 将原图图片通过Encoder编码得到高斯分布
            z, kld = Enc(real_input_pic0)
            # 将高斯分布反向解码生成抽样图片及类别合成的数据，记为sample_input
            sample_input = Gen(z)
            # 随机生成高斯分布的噪声输入Generator得到生成数据，记为generate_input
            rand_noise = (torch.rand(batch_size, noise_dim) - 0.5) * 2
            generate_input = Gen(Variable(rand_noise).to(device=DEVICE))

            # 将三种数据都放入判别器
            real_score = Dis(real_input)
            sample_score = Dis(sample_input)
            generate_score = Dis(generate_input)

            ############################
            # (1) Update D network
            ############################
            loss_D = lossD(real_score, sample_score, generate_score)
            D_trainer.zero_grad()
            loss_D.backward()
            D_trainer.step()

            ############################
            # (2) Update E&G network
            ###########################
            real_input_pic = data[0].to(device=DEVICE)  # 原图片bs*1*28*28
            real_input_pic0 = real_input_pic.view(batch_size, -1)   # 将原图摊平bs*784
            Target_score = Tar_CNN(real_input_pic)
            real_input = Combine_data(real_input_pic, Target_score) # 得到bs*1*(784+10)真值
            z, kld = Enc(real_input_pic0)
            sample_input = Gen(z)
            rand_noise = (torch.rand(batch_size, noise_dim) - 0.5) * 2
            generate_input = Gen(Variable(rand_noise).to(device=DEVICE))
            real_score = Dis(real_input)
            sample_score = Dis(sample_input)
            generate_score = Dis(generate_input)

            loss_GD = lossGD(generate_score)
            loss_G = 0.5 * (0.01*(sample_input - real_input).pow(2).sum()) / batch_size
            G_trainer.zero_grad()
            E_trainer.zero_grad()
            kld.backward(retain_graph=True)
            (0.01*loss_G+loss_GD).backward(torch.ones_like(loss_G))
            G_trainer.step()
            E_trainer.step()

            # Output training stats
            end_time = time.time()
            run_time = round(end_time-beg_time)


            print(
                f'Epoch: [{epoch+1:0>{len(str(num_epochs))}}/{num_epochs}]',
                f'Step: [{i+1:0>{len(str(len(dataloaders)))}}/{len(dataloaders)}]',
                f'Loss-D: {loss_D.item():.4f}',
                f'Loss-G: {(0.01*loss_G+loss_GD).item():.4f}',
                f'Time: {run_time}s',
                end='\r'
            )

        if epoch % 2 == 0:
            imgs_numpy = deprocess_img(generate_input[:,:,0:784].reshape(batch_size,1,28,28).cpu().numpy())
            if not os.path.exists('./generate_img'):
                os.mkdir('./generate_img')
            show_images(imgs_numpy[0:16], './generate_img/image_{}.png'.format(epoch + 1))


