#!/bin/bash

export PYTHONPATH=$(dirname $(pwd)):${PYTHONPATH}

JOB_NAME=hello_world
GPUS_PER_NODE=1
GPUS=${GPUS_PER_NODE}

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    --partition=shlab_adg \
    python3 ../main/test_baseline_evaluate.py \
      --cfg_dir ../utils/config/samples/sample_carla_improve \