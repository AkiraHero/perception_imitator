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

import warnings
warnings.filterwarnings("ignore")

class BaselineInterface:
    def __init__(self, ckpt_file, cfg_dir):
        self._config = Configuration()
        self._cfg_dir = cfg_dir
        self.update_config()
        self._device = self._config.training_config["device"]  

        self._model = ModelFactory.get_model(self._config.model_config)
        self.set_model(ckpt_file)

        self._dataset = DatasetFactory.get_dataset(self._config.dataset_config)
        self._data_loader = self._dataset.get_data_loader()

    def update_config(self):
        args = self._config.get_shell_args_train()
        args.cfg_dir = self._cfg_dir
        args.for_train = False
        args.shuffle = False
        self._config.load_config(args.cfg_dir)
        self._config.overwrite_config_by_shell_args(args)

    def set_model(self, checkpoint_file):
        paras = torch.load(checkpoint_file)
        self._model.load_model_paras(paras)
        self._model.set_decode(True)
        self._model.set_eval()
        self._model.set_device(self._device)

    def get_test_input(self, idx):
        data = self._data_loader.dataset[idx]
    
        occupancy = torch.from_numpy(data['occupancy']).permute(2, 0, 1)
        occlusion = torch.from_numpy(data['occlusion']).permute(2, 0, 1)
        HDmap = torch.from_numpy(data['HDmap']).permute(2, 0, 1)

        # get input
        input = torch.cat((occupancy, occlusion, HDmap), dim=0).float().to(self._device)

        return input

    def __call__(self, idx):
        with torch.no_grad():
            self._test_input = self.get_test_input(idx)

            # Forward Detection
            pred, features = self._model(self._test_input.unsqueeze(0))
            pred.squeeze_(0)

            corners, scores = filter_pred(self._config, pred)

            return corners