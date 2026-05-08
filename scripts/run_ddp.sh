#!/bin/bash

LAUNCH_DIR="/hzztai"

##### define user args
CONFIG_FILE="$1"
USER_ARGS=("${@:2}")

##### check args
if [ -z "${CONFIG_FILE}" ]; then
  echo "Error: CONFIG_FILE is not provided"
  exit 1
fi

##### Add working path to PYTHONPATH
TOOLBOX_ROOT_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"
export PYTHONPATH="${TOOLBOX_ROOT_DIR}":"${PYTHONPATH}"

##### resolve accelerator from args/env/runtime
source "${TOOLBOX_ROOT_DIR}/scripts/select_accelerator.sh"
select_accelerator "${USER_ARGS[@]}" || exit 1
DEVICE="${SELECT_ACCELERATOR_DEVICE}"
DEVICE_COUNT="${SELECT_ACCELERATOR_DEVICE_COUNT}"
if [ "${SELECT_ACCELERATOR_APPEND_DEVICE}" = "1" ]; then
  USER_ARGS+=("--device" "${DEVICE}")
fi

##### set environment variables for distributed training

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
if [ "${DEVICE}" = "cuda" ]; then
  export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
  export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
  export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
  export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}
elif [ "${DEVICE}" = "npu" ]; then
  export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}
  export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
  export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
else
  export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
fi

cd ${LAUNCH_DIR}
echo "Entering directory: ${LAUNCH_DIR}"

##### define nodes number for jobs
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-${DEVICE_COUNT}}

##### random DDP info
free_port=$(python -c "from slimai.helper.utils.dist_env import get_dist_env; print(get_dist_env().get_free_port())")
if [ -z "${MASTER_ADDR}" ]; then
  if [ "${NNODES}" = "1" ]; then
    MASTER_ADDR="127.0.0.1"
  else
    MASTER_ADDR="localhost"
  fi
fi
MASTER_PORT=${MASTER_PORT:-${free_port}}
MAX_RESTARTS=${MAX_RESTARTS:-3} # 0 for no restart, set default to 3 for 3 times restart
JOB_ID=${JOB_ID:-${MASTER_PORT}}

##### print job info
echo """<<< JOB SUMMARY >>>
Python Interpreter: $(which python)
CONFIG_FILE: ${CONFIG_FILE}
CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}
ASCEND_VISIBLE_DEVICES: ${ASCEND_VISIBLE_DEVICES:-N/A}
NNODES: ${NNODES}
NODE_RANK: ${NODE_RANK}
NPROC_PER_NODE: ${NPROC_PER_NODE}
DEVICE_COUNT: ${DEVICE_COUNT}
DEVICE: ${DEVICE}
JOB_ID: ${JOB_ID}
MASTER_ADDR: ${MASTER_ADDR}
MASTER_PORT: ${MASTER_PORT}
MAX_RESTARTS: ${MAX_RESTARTS}
===================
"""

TORCHRUN_ARGS=(
  --nnodes="${NNODES}"
  --node-rank="${NODE_RANK}"
  --nproc-per-node="${NPROC_PER_NODE}"
  --max-restarts="${MAX_RESTARTS}"
)

if [ "${NNODES}" = "1" ]; then
  echo "Running in static single-node mode"
  TORCHRUN_ARGS+=(
    --master-addr="${MASTER_ADDR}"
    --master-port="${MASTER_PORT}"
    --local-addr="${MASTER_ADDR}"
  )
else
  echo "Running in elastic multi-node mode"
  TORCHRUN_ARGS+=(
    --rdzv-id="${JOB_ID}"
    --rdzv-backend=c10d
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}"
  )
fi

##### DDP RUN
torchrun \
  "${TORCHRUN_ARGS[@]}" \
  ${TOOLBOX_ROOT_DIR}/tools/run.py \
  --config="${CONFIG_FILE}" \
  "${USER_ARGS[@]}"
