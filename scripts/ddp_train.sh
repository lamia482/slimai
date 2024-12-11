#!/bin/bash

CUDA_VISIBLE_DEVICES="5,6"

JOB_ID=12345
MASTER_ADDR=localhost
MASTER_PORT=12345

torchrun \
  --nnode=1 \
  --nproc-per-node=2 \
  --max-restarts=0 \
  --rdzv-id=$JOB_ID \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  train.py \
  --config=/hzztai/toolbox/_debug/debug.yml