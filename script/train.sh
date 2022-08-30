#!/bin/bash
srun -p shlab_adg -N 1 --gres=gpu:1 --quotatype=spot python main/train.py --cfg_dir utils/config/samples/sample_nuscenes/
