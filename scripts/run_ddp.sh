#!/bin/bash

##### define user args
CONFIG_FILE="$1"
CUDA_VISIBLE_DEVICES="7"

##### define nodes number for jobs
NNODES=1
NPROC_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())")

##### random DDP info
JOB_ID=12345
MASTER_ADDR=localhost
MASTER_PORT=12345
MAX_RESTARTS=3

##### print job info
echo """ JOB SUMMARY
CONFIG_FILE: ${CONFIG_FILE}
NNODES: ${NNODES}
NPROC_PER_NODE: ${NPROC_PER_NODE}
JOB_ID: ${JOB_ID}
MASTER_ADDR: ${MASTER_ADDR}
MASTER_PORT: ${MASTER_PORT}
MAX_RESTARTS: ${MAX_RESTARTS}
"""

##### cd working path
TOOLBOX_ROOT_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"
export PYTHONPATH="${TOOLBOX_ROOT_DIR}":"${PYTHONPATH}"

##### DDP RUN
torchrun \
  --nnode=${NNODES} \
  --nproc-per-node=${NPROC_PER_NODE} \
  --max-restarts=${MAX_RESTARTS} \
  --rdzv-id=${JOB_ID} \
  --rdzv-backend=c10d \
  --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  ${TOOLBOX_ROOT_DIR}/tools/run.py \
  --config="${CONFIG_FILE}" \
  --action="train" \
  ${@:2}
