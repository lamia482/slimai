#!/bin/bash

##### define user args
CONFIG_FILE="$1"

##### define nodes number for jobs
export NNODES=1
export NODE_RANK=${NODE_RANK:-0}

free_port=$(python -c "from slimai.helper.utils.dist_env import get_dist_env; print(get_dist_env().get_free_port())")

##### random DDP info
export MASTER_ADDR=10.168.100.88
export MASTER_PORT=${free_port}
export MAX_RESTARTS=0 # in multi-node training, set to 0 for no restart

if [ ${NNODES} -eq 1 ]; then
  export MASTER_ADDR=127.0.0.1
fi

# Call run_ddp.sh with the same arguments
bash "$(dirname "$(readlink -f "$0")")/run_ddp.sh" "${CONFIG_FILE}" ${@:2}
