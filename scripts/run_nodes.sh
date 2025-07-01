#!/bin/bash

##### define user args
CONFIG_FILE="$1"

##### define nodes number for jobs
export NNODES=1
export NODE_RANK=0

##### random DDP info
export JOB_ID=12345
export MASTER_ADDR=10.168.100.88
export MASTER_PORT=29603
export MAX_RESTARTS=1

if [ ${NNODES} -eq 1 ]; then
  export MASTER_ADDR=127.0.0.1
fi

# Call run_ddp.sh with the same arguments
bash "$(dirname "$(readlink -f "$0")")/run_ddp.sh" "${CONFIG_FILE}" ${@:2}
