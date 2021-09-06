#!/bin/bash

out_green(){
  echo -e "\033[32m $1 \033[0m"
}
export PYTHONPATH=$(dirname $(pwd)):${PYTHONPATH}

LOG_NAME=../output/ms_train_log-$(date "+%Y-%m-%d-%H-%M-%S").log
out_green "Set PYTHONPATH=${PYTHONPATH}"
out_green "Screen log will be redirected to file: ${LOG_NAME}"
python3 ../main/train.py\
  --cfg_dir ../utils/config/samples/sample_pvrcnn\
  2>&1 | tee "${LOG_NAME}"