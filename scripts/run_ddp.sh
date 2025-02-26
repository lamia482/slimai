#!/bin/bash

##### define user args
CONFIG_FILE="$1"

if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
  GPU_NUM="nvidia-smi -L | grep -c "UUID""
  export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($(eval $GPU_NUM)-1)))
fi

# export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

##### define nodes number for jobs
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
NPROC_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())")

##### random DDP info
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-0}
MAX_RESTARTS=${MAX_RESTARTS:-0}
JOB_ID=${JOB_ID:-${MASTER_PORT}}

##### check args
if [ -z "${CONFIG_FILE}" ]; then
  echo "Error: CONFIG_FILE is not provided"
  exit 1
fi

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
if  [ "${NNODES}" -eq 1 ]; then
  STANDALONE="--standalone"; \
  echo "Running in standalone mode"
fi

##### cd working path
TOOLBOX_ROOT_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"
export PYTHONPATH="${TOOLBOX_ROOT_DIR}":"${PYTHONPATH}"

##### DDP RUN
torchrun ${STANDALONE}\
  --nnodes=${NNODES} \
  --node-rank=${NODE_RANK} \
  --nproc-per-node=${NPROC_PER_NODE} \
  --max-restarts=${MAX_RESTARTS} \
  --rdzv-id=${JOB_ID} \
  --rdzv-backend=c10d \
  --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  ${TOOLBOX_ROOT_DIR}/tools/run.py \
  --config="${CONFIG_FILE}" \
  ${@:2}
