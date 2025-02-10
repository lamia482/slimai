#!/bin/bash

##### define user args
CONFIG_FILE="$1"
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"

##### define nodes number for jobs
NNODES=1
NODE_RANK=0
NPROC_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())")

##### random DDP info
JOB_ID=12345
MASTER_ADDR=localhost
MASTER_PORT=12345
MAX_RESTARTS=1

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
