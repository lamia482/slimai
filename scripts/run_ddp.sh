#!/bin/bash

##### define user args
CONFIG_FILE="$1"
CUDA_VISIBLE_DEVICES="5,6"

##### define nodes number for jobs
NNODES=1
NPROC_PER_NODE=2

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
cd "${TOOLBOX_ROOT_DIR}"
echo "Entering Root Working Path: ${TOOLBOX_ROOT_DIR}"

##### DDP RUN
torchrun \
  --nnode=${NNODES} \
  --nproc-per-node=${NPROC_PER_NODE} \
  --max-restarts=${MAX_RESTARTS} \
  --rdzv-id=${JOB_ID} \
  --rdzv-backend=c10d \
  --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  tools/run.py \
  --launcher=pytorch \
  --config="${CONFIG_FILE}" \
  --action="train" \
  --amp --auto-scale-lr \
  ${@:2}
