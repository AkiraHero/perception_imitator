'''
暂时用这一版
在原先GAN的基础上引入VAE的思想，此代码尝试VAE：从 源图像 编码-> 隐向量 采样解码-> 10个类别得分
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

# 定义画图工具,实际没有用到
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
        
    img_Din = data.squeeze().reshape((data.shape[0],1,pic_len)).to(device=DEVICE)   # 变为batch_size*1*len
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
    #     transform=transforms.Compose([
    #         # transforms.Resize(image_size),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5,), (0.5,))
    #     ]),
    #     download=True
    # )
    # dataset = train_data+test_data
    print(f'Total Size of Dataset: {len(dataset)}')

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers
    )
    return dataloader

def Load_Mnist_proc(): # 为了便于将图片压缩，直接使用datasets.MNIST中的transform
    train_data = datasets.MNIST( # train_set
        root=dataroot,
        train=True,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        download=True   # 首次使用设为True来下载数据集，之后设为False
    )
    dataset = train_data
    # test_data = datasets.MNIST( # test_set
    #     root=dataroot,
    #     train=False,
    #     transform=transforms.Compose([
    #         transforms.Resize(image_size),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5,), (0.5,))
    #     ]),
    #     download=True
    # )
    # dataset = train_data+test_data
    print(f'Total Size of Dataset: {len(dataset)}')

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers
    )
    return dataloader

# 定义VAE中编码器，将数据进行编码并重参化为mean和var（将其作为Generator）
class VAE(nn.Module):
    def __init__(self, ngpu):
        super(VAE, self).__init__()
        self.ngpu = ngpu
        self.fc1 = nn.Linear(784, 800)
        self.fc2 = nn.Linear(800, 400)
        self.fc21 = nn.Linear(400, 200) # mean
        self.fc22 = nn.Linear(400, 200) # var
        self.fc3 = nn.Linear(200, 400)
        self.fc4 = nn.Linear(400, 200)
        self.fc5 = nn.Linear(200, 10)

    def encode(self, x):    # 编码层
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc21(h2), self.fc22(h2)

    def reparametrize(self, mu, logvar):    # 最后得到的是u(x)+sigma(x)*N(0,I)
        std = logvar.mul(0.5).exp_() # e**(x/2)
        eps = torch.FloatTensor(std.size()).normal_()   # 用正态分布填充eps
        if torch.cuda.is_available():
            eps = Variable(eps.cuda())
        else:
            eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):    # 解码层
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc4(h3))
        return self.fc5(h4)

    def forward(self, x):
        mu, logvar = self.encode(x) # 编码
        z = self.reparametrize(mu, logvar) # 重新参数化成正态分布
        return self.decode(z), mu, logvar # 解码，同时输出均值方差

class Discriminator_MLP(nn.Module):
    def __init__(self, ngpu, input_size):
        super(Discriminator_MLP, self).__init__()
        self.ngpu = ngpu
        # 初始化四层神经网络 两个全连接的隐藏层，一个输出层
        self.fc1 = nn.Linear(input_size,1000) # 第一个隐含层
        self.fc2 = nn.Linear(1000,1000) # 第二个隐含层
        self.fc3 = nn.Linear(1000,500) # 第三个隐含层
        self.fc4 = nn.Linear(500,1)  # 输出层
        
        self.dropout = nn.Dropout(p=0.5)    # Dropout暂时先不打开

    def forward(self, din):
        # 前向传播， 输入值：din, 返回值 dout
        din = din.view(-1,din.shape[2])    # 将一个多行的Tensor,拼接成一行
        dout = self.fc1(din)
        dout = F.relu(dout)  # 使用 relu 激活函数
        dout = self.fc2(dout)
        dout = F.relu(dout)
        dout = self.fc3(dout)
        dout = F.relu(dout)
        dout = self.fc4(dout)
        dout = F.sigmoid(dout)
        # dout = F.softmax(dout, dim=1) # 输出层使用 softmax 激活函数,这里输出层为1,因此不需要softmax,使用sigmoid

        return dout

def discriminator_loss(scores_real, scores_fake): #判别器的loss
    loss = ((scores_real - 1) ** 2).mean() + (scores_fake ** 2).mean()
    return loss

def generator_loss(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    loss0 = ((recon_x - 1) ** 2).mean()
    # KL divergence
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return loss0+KLD

def get_optimizer(net):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.00015, betas=(0.5, 0.999))
    return optimizer

if __name__ == '__main__':
    dataroot = "/home/PJLAB/sunyiyang/桌面/PJlab/GAN_Exp/Datasets"  # Root directory for dataset
    dataroot = "/home/PJLAB/sunyiyang/桌面/PJlab/GAN_Exp/Datasets"  # Root directory for dataset
    workers = 10    # Number of workers for dataloader
    batch_size = 100    # Batch size during training
    image_size = 28  # 可以根据自己的需求改变，这里把图像缩成14*14个像素，Spatial size of training images. All images will be resized to this size using a transformer.
    class_size = 10  # 分为十类
    input_size = image_size*image_size + class_size
    nc = 1          # Number of channels in the training images. For color images this is 3
    num_epochs = 100 # Number of training epochs
    lr = 0.0002     # Learning rate for optimizers
    beta1 = 0.3     # Beta1 hyperparam for Adam optimizers
    ngpu = 1        # Number of GPUs available. Use 0 for CPU mode.

    DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')

    dataloader1 = Load_Mnist_ori()
    dataloader2 = Load_Mnist_proc()
    
    # 加载Target CNN模型，用于得到参照的真实分类
    Tar_CNN= Net().to(device=DEVICE)
    Tar_CNN.load_state_dict(torch.load('./results/Mnist/param_minist_5.pt'))

    # 建立生成器、判别器
    netG = VAE(ngpu).to(device=DEVICE)   # Create the generator
    if DEVICE.type == 'cuda' and ngpu > 1:  # Handle multi-gpu if desired
        netG = nn.DataParallel(netG, list(range(ngpu)))
    netG.apply(weights_init)    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netDMLP = Discriminator_MLP(ngpu, input_size).to(device=DEVICE)    # Create the Discriminator
    if DEVICE.type == 'cuda' and ngpu > 1:
        netDMLP = nn.DataParallel(netDMLP, list(range(ngpu)))
    netDMLP.apply(weights_init)

    # 初始化优化器
    G_optimizer = get_optimizer(netG)
    D_optimizer = get_optimizer(netDMLP)
    
    # 初始化Optimizers和损失函数
    criterion = nn.BCELoss()    # Initialize BCELoss function
    # 方便建立真值，Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0.
    True_label = torch.full((batch_size,), real_label, dtype=torch.float, device=DEVICE)
    Fake_label = torch.full((batch_size,), fake_label, dtype=torch.float, device=DEVICE)

    # # Setup Adam optimizers for both G and D
    # optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    # optimizerDMLP = torch.optim.Adam(netDMLP.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    D_x_list = []
    D_z_list = []
    loss_tep1 = 10
    loss_tep2 = 10
    iter_count = 0
    # min = -0.05
    # max = 0.05

    # 开始训练
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        beg_time = time.time()
        for i, data in enumerate(zip(dataloader1, dataloader2)): # dataloader_ori存放原图
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            netDMLP.zero_grad()
            # real_cpu = Change_data(data[1]).to(device=DEVICE)   # 将压缩图像转为能够输入D的batch_size*1*(img_size*img_size+1)的数据
            # 把原图像经过Target_CNN网络的label再结合原图作为真值
            G_input = data[0][0].to(device=DEVICE)
            Target_score = Tar_CNN(G_input)
            # _, Target_score = torch.max(Target_score.data, 1)
            real_cpu =  Combine_data(data[1][0], Target_score).to(device=DEVICE)

            # Forward pass real batch through D
            logits_real = netDMLP(real_cpu).view(-1)

            d_total_error_1 = criterion(logits_real, True_label).mul_(0.5)
            d_total_error_1.backward()

            # Train with all-fake batch
            G_input = data[0][0].view(batch_size, -1).to(device=DEVICE)
            # Generate fake image batch with G
            G_score, mu, logvar = netG(G_input)    # 通过VAE得到每一类的得分
            # _, G_score = torch.max(G_score.data, 1)
            # 处理压缩图和G生成类别得到可用于输入D的数据
            fake = Combine_data(data[1][0], G_score).to(device=DEVICE)
            # Classify all fake batch with D
            logits_fake = netDMLP(fake.detach()).view(-1)

            d_total_error_2 = criterion(logits_fake, Fake_label).mul_(0.5)
            d_total_error_2.backward()
            
            # d_total_error = discriminator_loss(logits_real, logits_fake) # 判别器的 loss
            d_total_error = d_total_error_1 + d_total_error_2

            # D_optimizer.zero_grad()
            # netDMLP.zero_grad()
            # d_total_error.backward()
            D_optimizer.step() # 优化判别网络

            # # 用来限制D网络参数范围
            # for p in netDMLP.parameters():
            #     p.data.clamp_(min, max)


            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()

            # G_score, mu, logvar = netG(G_input)
            # fake = Combine_data(data[1][0], G_score).to(device=DEVICE)
            gen_logits_fake = netDMLP(fake).view(-1)

            g_error = criterion(gen_logits_fake, True_label)
            KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            KLD = F.sigmoid(torch.sum(KLD_element).mul_(-0.5))  # 可能限制一下KLD效果会变好
            g_error = g_error.add_(KLD)

            # g_error = generator_loss(gen_logits_fake, G_input, mu, logvar)
            # G_optimizer.zero_grad()

            # netG.zero_grad()
            g_error.backward()
            G_optimizer.step() # 优化生成网络

            if (iter_count % 300 == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.item(), g_error.item()))
            iter_count += 1

            # Save the Best Model
            if g_error < loss_tep1 and epoch > 10:
                torch.save(netG.state_dict(), './results/VAE_Mnist/model_errG.pt')
                loss_tep1 = g_error
            if epoch % 10 == 0:  
                torch.save(netG.state_dict(), './results/VAE_Mnist/model_%d.pt'%(epoch))
            # # # Show part results
            # # if epoch % 2 == 0:
            # #     imgs_numpy = deprocess_img(generate_input[:,:,0:784].reshape(batch_size,1,28,28).cpu().numpy())
            # # if not os.path.exists('./generate_img'):
            # #     os.mkdir('./generate_img')
            # # show_images(imgs_numpy[0:16], './generate_img/image_{}.png'.format(epoch + 1))


    torch.save(netG.state_dict(), './results/VAE_Mnist/model_final.pt')

    # plt.figure(figsize=(20, 10))
    # plt.title("Generator and Discriminator Loss During Training")
    # plt.plot(G_losses[::100], label="G")
    # plt.plot(D_losses[::100], label="D")
    # plt.xlabel("iterations")
    # plt.ylabel("Loss")
    # plt.axhline(y=0, label="0", c="g") # asymptote
    # plt.legend()
    # plt.show()

    # plt.figure(figsize=(20, 10))
    # plt.title("D(x) and D(G(z)) During Training")
    # plt.plot(D_x_list[::100], label="D(x)")
    # plt.plot(D_z_list[::100], label="D(G(z))")
    # plt.xlabel("iterations")
    # plt.ylabel("Probability")
    # plt.axhline(y=0.5, label="0.5", c="g") # asymptote
    # plt.legend()
    # plt.show()

