'''
在原先GAN的基础上引入VAE的思想，此代码尝试VAE：从 源图像 编码-> 隐向量 采样解码-> 10个类别得分
v5:直接载GAN_Mnist_10上进行改动
'''
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utils.logger.basic_logger import getLogger



if __name__ == '__main__':
    logger = getLogger(__name__)
    dataroot = "/home/PJLAB/sunyiyang/桌面/PJlab/GAN_Exp/Datasets"  # Root directory for dataset
    workers = 10    # Number of workers for dataloader
    batch_size = 100    # Batch size during training
    image_size = 14  # 可以根据自己的需求改变，这里把图像缩成14*14个像素，Spatial size of training images. All images will be resized to this size using a transformer.
    class_size = 10  # 分为十类
    input_size = image_size*image_size + class_size
    nc = 1          # Number of channels in the training images. For color images this is 3
    num_epochs = 100 # Number of training epochs
    lr = 0.0001     # Learning rate for optimizers
    beta1 = 0.5     # Beta1 hyperparam for Adam optimizers
    ngpu = 1        # Number of GPUs available. Use 0 for CPU mode.
    
    DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
    
    # 加载数据
    # dataloader_ori = Load_Mnist_ori()
    # dataloader_proc = Load_Mnist_proc()
    from data_loader.minist_loader import MinistDataset
    data_loader = MinistDataset.get_data_loader_instance()
    # get compressed img here
    from torchvision import transforms

    transform_proc = transforms.Compose([transforms.Resize(image_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))])


    # 加载Target CNN模型
    from model.basic_module.cnn import CNN
    from model.basic_module.vae import VAE
    from model.basic_module.mlp import MLP
    from utils.model.model_utils import weights_init, Combine_data

    Tar_CNN = CNN().to(device=DEVICE)
    Tar_CNN.load_state_dict(torch.load('./results/Mnist/param_minist_5.pt'))

    # 建立生成器、判别器
    netG = VAE().to(device=DEVICE)   # Create the generator
    if DEVICE.type == 'cuda' and ngpu > 1:  # Handle multi-gpu if desired
        netG = nn.DataParallel(netG, list(range(ngpu)))
    netG.apply(weights_init)    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netDMLP = MLP(ngpu, input_size).to(device=DEVICE)    # Create the Discriminator
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
        # For each batch in the dataloader
        for i, data in enumerate(zip(data_loader)): # dataloader_ori存放原图，用于训练G；dataloader_proc存放缩减图，处理后用于训练D
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netDMLP.zero_grad()
            # Format batch
            
            # real_cpu = Change_data(data[1]).to(device=DEVICE)   # 将压缩图像转为能够输入D的batch_size*1*(img_size*img_size+1)的数据
            # 把原图像经过Target_CNN网络的label结合压缩图作为真值
            G_input = data[0].to(device=DEVICE)
            Target_score = Tar_CNN(G_input)
            # _, Target_score = torch.max(Target_score.data, 1)


            compressed_data = transform_proc.__call__(data[0])


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
            G_score, mu, logvar = netG(G_input)    # 得到每一类的得分
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
            errG1 = criterion(output, label)     # 希望生成的假数据能让D判成1
            # Calculate gradients for G
            # errG.backward()

            KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            errG_KLD = F.sigmoid(torch.sum(KLD_element))

            errG = errG1.add_(errG_KLD)
            errG.backward()

            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            end_time = time.time()
            run_time = round(end_time-beg_time)

            cur_info ="\n".join([f'Epoch: [{epoch+1:0>{len(str(num_epochs))}}/{num_epochs}]'
                f'Step: [{i+1:0>{len(str(len(data_loader)))}}/{len(data_loader)}]',
                f'Loss-D: {errD.item():.4f}',
                f'Loss-G: {errG.item():.4f}',
                f'D(x): {D_x:.4f}',
                f'D(G(z)): [{D_G_z1:.4f}/{D_G_z2:.4f}]',
                f'Time: {run_time}s'])
            logger.info(cur_info)

            
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            # Save D(X) and D(G(z)) for plotting later
            D_x_list.append(D_x)
            D_z_list.append(D_G_z2)

            # Save the Best Model
            if errG < loss_tep1 and epoch > 10:
                torch.save(netG.state_dict(), '../results/VAE_Mnist2/model_errG.pt')
                loss_tep1 = errG
            if epoch%10 == 0:  
                torch.save(netG.state_dict(), './results/VAE_Mnist2/model_%d.pt'%(epoch))

    # todo： use logger
    torch.save(netG.state_dict(), '../results/VAE_Mnist2/model_final.pt')




    # todo: move utils-vis
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
