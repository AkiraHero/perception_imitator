import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.nn as nn
import numpy as np

class MinistDataset(Dataset):
    def __init__(self, config):
        self._is_train = config['FOR_TRAIN']
        self._data_root = config['DATA_ROOT']
        self._batch_size = config['BATCH_SIZE']
        self._shuffle = config['SHUFFLE']
        self._num_workers = config['NUM_WORKERS']
        self._embeding_dataset = datasets.MNIST(  # train_set
            root=self._data_root,
            train=self._is_train,
            transform=transforms.ToTensor(),
            download=False
        )

    def __len__(self):
        return len(self._embeding_dataset)

    def __getitem__(self, item):
        assert item <= self.__len__()
        return self._embeding_dataset[item]

    @staticmethod
    def get_data_loader_instance():
        dataset = MinistDataset()
        return DataLoader(
            dataset=dataset,
            batch_size=dataset._batch_size,
            shuffle=dataset._shuffle,
            num_workers=dataset._num_workers
        )


#
#
# def Load_Mnist_proc():  # 为了便于将图片压缩，直接使用datasets.MNIST中的transform
#     train_data = datasets.MNIST(
#         root=dataroot,
#         train=True,
#         transform=transforms.Compose([
#             transforms.Resize(image_size),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,))
#         ]),
#         download=True
#     )
#     test_data = datasets.MNIST(
#         root=dataroot,
#         train=False,
#         transform=transforms.Compose([
#             transforms.Resize(image_size),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,))
#         ])
#     )
#     dataset = train_data + test_data
#     # print(f'Total Size of Dataset: {len(dataset)}')
#
#     dataloader = DataLoader(
#         dataset=dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=workers
#     )
#
#     return dataloader


# def Change_data(data):  # 对真值图片和标签进行处理(之前几版代码见pytorch-dcgan-mnist)
#     pic_len = data[0].shape[1] * data[0].shape[2] * data[0].shape[3]  # 获取每一张图片压缩后的总像素个数
#
#     img_Din = data[0].numpy().squeeze().reshape((data[0].shape[0], 1, pic_len))  # 变为batch_size*1*16
#     label_Din = data[1].unsqueeze(-1).unsqueeze(-1).numpy()  # 获得对应label
#     img_Din = torch.from_numpy(np.append(img_Din, label_Din, axis=2))  # 将label加入得到batch_size*1*17，并转为tensor类型
#     img_Din = img_Din.to(torch.float32)  # 将double精度转为float适用于全连接层输入类型
#     # print(img_Din.shape)
#
#     return img_Din
#
#
# def Combine_data(data, label):  # 直接对tensor类型进行处理，这样可以保存反传的梯度，将处理后图片与经过G得到的类别组合成可以输入D的数据
#     pic_len = data.shape[1] * data.shape[2] * data.shape[3]  # 获取每一张图片压缩后的总像素个数
#
#     img_Din = data.squeeze().reshape((data.shape[0], 1, pic_len))  # 变为batch_size*1*len
#     # label_Din = label.cpu().unsqueeze(-1).unsqueeze(-1).numpy()   # 获得对应label
#     label_Din = label.cpu().unsqueeze(-2)  # 获得对应label,对于增添10各类别概率仅需要加一个维度即可
#     img_Din = torch.cat((img_Din, label_Din), 2)  # 将label余图像直接tensor合并，得到batch_size*1*(len+10)，主要为了是的tensor能够用保留反传梯度
#     # img_Din = torch.from_numpy(np.append(img_Din, label_Din, axis=2)) # 将label加入得到batch_size*1*(len+10)
#     # img_Din = img_Din.to(torch.float32) # 将double精度转为float适用于全连接层输入类型
#
#     return img_Din





# dataloader_ori = Load_Mnist_ori()
# dataloader_proc = Load_Mnist_proc()

# def get_data_loader:
#
#     pass