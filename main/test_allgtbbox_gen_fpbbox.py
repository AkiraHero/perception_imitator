import sys
sys.path.append('D:/1Pjlab/ADModel_Pro/')
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from utils.config.Configuration import Configuration
from utils.IoU_calculate import boxes_iou3d_cpu
from factory.model_factory import ModelFactory
from factory.dataset_factory import DatasetFactory
from collections import OrderedDict

def list_of_tensor_2_tensor(data):
    data = torch.stack(data, 0)
    data = data.cpu()

    return data

def change_data_form(data):
    for k, v in data.items():
        if k in ['gt_bboxes', 'fp_bboxes', 'difficult']:
            v = torch.stack(v, 0)
            v = v.transpose(0,1).to(torch.float32)
            data[k] = v
        else:
            data[k] = v

if __name__ == '__main__':
    # manage config
    config = Configuration()
    args = config.get_shell_args_train()
    # args.batch_size = 1024
    args.for_train = False
    args.shuffle = False
    config.load_config(args.cfg_dir)
    config.overwrite_config_by_shell_args(args)

    # instantiating all modules by non-singleton factory
    model = ModelFactory.get_model(config.model_config)

    paras = torch.load("D:/1Pjlab/ADModel_Pro/output/gtbbox_gen_fpbbox_model/660.pt")
    model.generator.load_model_paras(paras)
    model.set_eval()
    model.set_device("cuda:0")
    dataset = DatasetFactory.get_dataset(config.dataset_config)
    data_loader = dataset.get_data_loader()

    gt_fp_bbox = []
    gen_fp_bbox = []
    gt_fp_difficult= []

    with torch.no_grad():
        for step, data in enumerate(data_loader):
            change_data_form(data)

            print('Step:', step)
            generate_input = data['gt_bboxes'].cuda()
            dt_box_fp = data['fp_bboxes'].cuda()
            difficult = data['difficult'].cuda()

            gen_data,_,_ = model.generator(generate_input)
            dt_box_fp = dt_box_fp.view(dt_box_fp.shape[0], -1, 7)
            gen_data = gen_data.view(dt_box_fp.shape[0], -1, 7)

            gt_fp_bbox.extend(dt_box_fp)
            gen_fp_bbox.extend(gen_data)
            gt_fp_difficult.extend(difficult)
    
    gt_fp_bbox = list_of_tensor_2_tensor(gt_fp_bbox)
    gen_fp_bbox = list_of_tensor_2_tensor(gen_fp_bbox)
    print(gt_fp_bbox.shape)
    print(gen_fp_bbox.shape)

    all_iou = []
    # for i in range(gen_fp_bbox.shape[0]):
    for i in range(0,10):
        print(gen_fp_bbox[i], gt_fp_bbox[i])
        iou_3d = boxes_iou3d_cpu(gen_fp_bbox[i], gt_fp_bbox[i])
        all_iou.append(iou_3d)

    all_iou = list_of_tensor_2_tensor(all_iou)
    print("IoU_3D is:", all_iou)    # 目前高度的生成可能存在问题，导致3D检测框的IoU值为0，但是从bev视角看，是可以生成有重合部分的FP框的
    print("difficult:", gt_fp_difficult[0:10])

    pass