#!/bin/bash
set -x
out_green(){
  echo -e "\033[32m $1 \033[0m"
}
export PYTHONPATH=$(dirname $(pwd)):${PYTHONPATH}

LOG_NAME=../output/ms_train_log-$(date "+%Y-%m-%d-%H-%M-%S").log
out_green "Set PYTHONPATH=${PYTHONPATH}"
out_green "Screen log will be redirected to file: ${LOG_NAME}"

JOB_NAME=hello_world
GPUS_PER_NODE=1
GPUS=${GPUS_PER_NODE}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    --partition=shlab_adg \
    python3 ../main/train.py\
      --cfg_dir ../utils/config/samples/sample_pvrcnn\
      --screen_log ${LOG_NAME}\
      --distributed \
      --launcher slurm \
      --tcp_port $PORT \
      2>&1 | tee -i "${LOG_NAME}"
