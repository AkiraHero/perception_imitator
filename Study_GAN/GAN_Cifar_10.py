import sys
from torch._C import device
from torch.types import Device

from torch.utils import data
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import utils, datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
import cv2
from CNN_Cifar import Net

# Set random seed for reproducibility
torch.manual_seed(0)

def Load_Cifar_ori():
    train_data = datasets.CIFAR10( # train_set
        root=dataroot,
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        download=True   # 首次使用设为True来下载数据集，之后设为False
    )
    test_data = datasets.CIFAR10( # test_set
        root=dataroot,
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        download=True
    )
    dataset = train_data+test_data
    print(f'Total Size of Dataset: {len(dataset)}')

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers
    )

    return dataloader

def Load_Cifar_proc(): # 为了便于将图片压缩，直接使用datasets.MNIST中的transform
    train_data = datasets.CIFAR10(
        root=dataroot,
        train=True,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        download=True
        )
    test_data = datasets.CIFAR10(
        root=dataroot,
        train=False,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    dataset = train_data+test_data
    # print(f'Total Size of Dataset: {len(dataset)}')

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers
    )

    return dataloader

def Change_data(data):  # 对真值图片和标签进行处理(之前几版代码见pytorch-dcgan-mnist)
    pic_len = data[0].shape[1]*data[0].shape[2]*data[0].shape[3]    # 获取每一张图片压缩后的总像素个数
    
    img_Din = data[0].numpy().squeeze().reshape((data[0].shape[0],1,pic_len))   # 变为batch_size*1*length
    label_Din = data[1].unsqueeze(-1).unsqueeze(-1).numpy()   # 获得对应label
    img_Din = torch.from_numpy(np.append(img_Din, label_Din, axis=2)) # 将label加入得到batch_size*1*(length+10)，并转为tensor类型
    img_Din = img_Din.to(torch.float32) # 将double精度转为float适用于全连接层输入类型
    # print(img_Din.shape)

    return img_Din

# def Combine_data(data, label):     # 将处理后图片与经过G得到的类别组合成可以输入D的数据
#     pic_len = data.shape[1]*data.shape[2]*data.shape[3]    # 获取每一张图片压缩后的总像素个数
        
#     img_Din = data.numpy().squeeze().reshape((data.shape[0],1,pic_len))   # 变为batch_size*1*length
#     # label_Din = label.cpu().unsqueeze(-1).unsqueeze(-1).numpy()   # 获得对应label
#     label_Din = label.cpu().unsqueeze(-2).numpy()   # 获得对应label,对于增添10各类别概率仅需要加一个维度即可
#     img_Din = torch.from_numpy(np.append(img_Din, label_Din, axis=2)) # 将label加入得到batch_size*1*(length+10)
#     img_Din = img_Din.to(torch.float32) # 将double精度转为float适用于全连接层输入类型

#     return img_Din
    
def Combine_data(data, label):     # 直接对tensor类型进行处理，这样可以保存反传的梯度，将处理后图片与经过G得到的类别组合成可以输入D的数据
    pic_len = data.shape[1]*data.shape[2]*data.shape[3]    # 获取每一张图片压缩后的总像素个数（包括每个通道）= 通道数*宽*高
        
    img_Din = data.squeeze().reshape((data.shape[0],1,pic_len))   # 变为batch_size*1*len
    # label_Din = label.cpu().unsqueeze(-1).unsqueeze(-1).numpy()   # 获得对应label
    label_Din = label.cpu().unsqueeze(-2)   # 获得对应label,对于增添10各类别概率仅需要加一个维度即可
    img_Din = torch.cat((img_Din, label_Din), 2)    # 将label直接与图像tensor合并，得到batch_size*1*(len+10)
    # img_Din = torch.from_numpy(np.append(img_Din, label_Din, axis=2)) # 将label加入得到batch_size*1*(len+10)
    # img_Din = img_Din.to(torch.float32) # 将double精度转为float适用于全连接层输入类型

    return img_Din

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# 生成器结构（暂时直接从CNN_Mnist中调用Net，不用这个）
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
        )

    def forward(self, input):
        return self.main(input)

class Discriminator_MLP(nn.Module):
    def __init__(self, ngpu, input_size):
        super(Discriminator_MLP, self).__init__()
        self.ngpu = ngpu
        # 初始化四层神经网络 三个全连接的隐藏层，一个输出层
        self.fc1 = nn.Linear(input_size,600) # 第一个隐含层
        self.fc2 = nn.Linear(600,600) # 第二个隐含层
        self.fc3 = nn.Linear(600,200)  # 输出层
        self.fc4 = nn.Linear(200,1)  # 输出层
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

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 2 every 20 epochs"""
    lr = lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    dataroot = "/home/PJLAB/sunyiyang/桌面/PJlab/GAN_Exp/Datasets"  # Root directory for dataset
    workers = 10    # Number of workers for dataloader
    batch_size = 100    # Batch size during training
    image_size = 12  # 可以根据自己的需求改变，这里把图像缩成16个像素，Spatial size of training images. All images will be resized to this size using a transformer.
    class_size = 10  # 分为十类
    input_size = 3*image_size*image_size + class_size   # 需要考虑通道数3
    nc = 1          # Number of channels in the training images. For color images this is 3
    num_epochs = 70 # Number of training epochs
    lr = 0.0002  # Learning rate for optimizers
    beta1 = 0.5     # Beta1 hyperparam for Adam optimizers
    ngpu = 1        # Number of GPUs available. Use 0 for CPU mode.
    
    DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
    
    # 加载数据
    dataloader_ori = Load_Cifar_ori()
    dataloader_proc = Load_Cifar_proc()

    # 加载Target CNN模型
    Tar_CNN= Net().to(device=DEVICE)
    Tar_CNN.load_state_dict(torch.load('./results/Cifar/param_minist_10.pt'))

    # 建立生成器、判别器
    netG = Net().to(device=DEVICE)   # Create the generator
    if DEVICE.type == 'cuda' and ngpu > 1:  # Handle multi-gpu if desired
        netG = nn.DataParallel(netG, list(range(ngpu)))
    netG.apply(weights_init)    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netDMLP = Discriminator_MLP(ngpu, input_size).to(device=DEVICE)    # Create the Discriminator
    if DEVICE.type == 'cuda' and ngpu > 1:
        netDMLP = nn.DataParallel(netDMLP, list(range(ngpu)))
    netDMLP.apply(weights_init)

    # 初始化Optimizers和损失函数
    criterion = nn.BCELoss()    # Initialize BCELoss function
    # 方便建立真值，Establish convention for real and fake labels during training
    real_label = 1. 
    fake_label = 0.
    # Setup Adam optimizers for both G and D
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerDMLP = torch.optim.Adam(netDMLP.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    D_x_list = []
    D_z_list = []
    loss_tep1 = 10
    loss_tep2 = 10

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        beg_time = time.time()
        adjust_learning_rate(optimizerG, epoch, lr)
        adjust_learning_rate(optimizerDMLP, epoch, lr)
        # For each batch in the dataloader
        for i, data in enumerate(zip(dataloader_ori, dataloader_proc)): # dataloader_ori存放原图，用于训练G；dataloader_proc存放缩减图，处理后用于训练D
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netDMLP.zero_grad()
            # Format batch
            
            # real_cpu = Change_data(data[1]).to(device=DEVICE)   # 将压缩图像转为能够输入D的batch_size*1*(img_size*img_size+1)的数据
            # 把原图像经过Target_CNN网络的label结合压缩图作为真值
            G_input = data[0][0].to(device=DEVICE)
            Target_score = Tar_CNN(G_input)
            # _, Target_score = torch.max(Target_score.data, 1)
            real_cpu =  Combine_data(data[1][0], Target_score).to(device=DEVICE)
            b_size = real_cpu.size(0)   # batch_size值
            label = torch.full((b_size,), real_label, dtype=torch.float, device=DEVICE)

            # Forward pass real batch through D
            output = netDMLP(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()  


            # Train with all-fake batch
            G_input = data[0][0].to(device=DEVICE)
            # Generate fake image batch with G
            G_score = netG(G_input)    # 得到每一类的得分
            # _, G_score = torch.max(G_score.data, 1)
            # 处理压缩图和G生成类别得到可用于输入D的数据
            fake = Combine_data(data[1][0], G_score).to(device=DEVICE)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netDMLP(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()   
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake    # 希望对真实数据接近label1，对于假数据接近label0
            # Update D
            optimizerDMLP.step()


            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netDMLP(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)     # 希望生成的假数据能让D判成1
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            end_time = time.time()
            run_time = round(end_time-beg_time)

            print(
                f'Epoch: [{epoch+1:0>{len(str(num_epochs))}}/{num_epochs}]',
                f'Step: [{i+1:0>{len(str(len(dataloader_ori)))}}/{len(dataloader_ori)}]',
                f'Loss-D: {errD.item():.4f}',
                f'Loss-G: {errG.item():.4f}',
                f'D(x): {D_x:.4f}',
                f'D(G(z)): [{D_G_z1:.4f}/{D_G_z2:.4f}]',
                f'Time: {run_time}s',
                end='\r'
            )
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            # Save D(X) and D(G(z)) for plotting later
            D_x_list.append(D_x)
            D_z_list.append(D_G_z2)

            # Save the Best Model
            if errG < loss_tep1 and epoch > 10:
                torch.save(netG.state_dict(), './results/Cifar/model_errG.pt')
                loss_tep1 = errG
            if epoch%10 == 0:  
                torch.save(netG.state_dict(), './results/Cifar/model_%d.pt'%(epoch))

    torch.save(netG.state_dict(), './results/Cifar/model_final.pt')

    plt.figure(figsize=(20, 10))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses[::100], label="G")
    plt.plot(D_losses[::100], label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.axhline(y=0, label="0", c="g") # asymptote
    plt.legend()
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.title("D(x) and D(G(z)) During Training")
    plt.plot(D_x_list[::100], label="D(x)")
    plt.plot(D_z_list[::100], label="D(G(z))")
    plt.xlabel("iterations")
    plt.ylabel("Probability")
    plt.axhline(y=0.5, label="0.5", c="g") # asymptote
    plt.legend()
    plt.show()
