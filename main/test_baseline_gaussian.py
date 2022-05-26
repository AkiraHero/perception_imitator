import os
import sys
sys.path.append(os.getcwd())
import torch
from torch.multiprocessing import Pool
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import random
from factory.model_factory import ModelFactory
from factory.dataset_factory import DatasetFactory
from utils.config.Configuration import Configuration
from utils.loss import CustomLoss, SmoothL1Loss
from utils.postprocess import *
from utils.visualize import get_bev, plot_bev, plot_label_map
from collections import OrderedDict
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # manage config
    config = Configuration()
    args = config.get_shell_args_train()
    args.for_train = False
    args.shuffle = False
    config.load_config(args.cfg_dir)
    config.overwrite_config_by_shell_args(args)

    # instantiating loss and dataset
    perception_loss_func = CustomLoss(config.training_config['loss_function'])
    prediction_loss_func = SmoothL1Loss()
    dataset = DatasetFactory.get_dataset(config.dataset_config)
    data_loader = dataset.get_data_loader()

    # get GT
    index = 0
    data = data_loader.dataset[index]
    occupancy = data['occupancy']
    _, label_list, _, _, _ = dataset.get_label(index)

    ######################
    # Add Gaussian Noise #
    ######################
    gaussian_dist = [0, 0.2]    # Gaussian distribution
    fn_rate = 0.1              # Target model's false negative
    guassian_noise_list = dataset.get_gaussian_noise(index, mu=gaussian_dist[0], sigma=gaussian_dist[1], drop=fn_rate)

    # plot_bev(occupancy, guassian_noise_list, window_name='Gaussian')
    plot_bev(occupancy, label_list, window_name='DT')


    ########################
    # Add Multimodal Noise #
    ########################
    multimodal_noise_list =  dataset.get_multimodal_noise(index, drop=fn_rate)

    # plot_bev(occupancy, multimodal_noise_list, window_name='MultiModal')
    # plot_bev(occupancy, label_list, window_name='DT')
