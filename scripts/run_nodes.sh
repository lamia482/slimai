#!/bin/bash

NCCL_DEBUG=INFO

##### define user args
CONFIG_FILE="$1"
# CUDA_VISIBLE_DEVICES="3,4"

##### define nodes number for jobs
NNODES=1
NODE_RANK=0
NPROC_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())")

##### random DDP info
JOB_ID=12345
MASTER_ADDR=10.168.100.88
MASTER_PORT=29603
MAX_RESTARTS=1

##### print job info
echo """<<< JOB SUMMARY >>>
CONFIG_FILE: ${CONFIG_FILE}
NNODES: ${NNODES}
NODE_RANK: ${NODE_RANK}
NPROC_PER_NODE: ${NPROC_PER_NODE}
JOB_ID: ${JOB_ID}
MASTER_ADDR: ${MASTER_ADDR}
MASTER_PORT: ${MASTER_PORT}
MAX_RESTARTS: ${MAX_RESTARTS}
===================
"""

if [ ${NODE_RANK} -eq 0 ]; then
  MASTER_ADDR=127.0.0.1
fi

##### cd working path
TOOLBOX_ROOT_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"
export PYTHONPATH="${TOOLBOX_ROOT_DIR}":"${PYTHONPATH}"

##### DDP RUN
torchrun \
  --nnodes=${NNODES} \
  --node-rank=${NODE_RANK} \
  --nproc-per-node=${NPROC_PER_NODE} \
  --max-restarts=${MAX_RESTARTS} \
  --rdzv-id=${JOB_ID} \
  --rdzv-backend=c10d \
  --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  ${TOOLBOX_ROOT_DIR}/tools/run.py \
  --config="${CONFIG_FILE}" \
  --action="train" \
  ${@:2}
