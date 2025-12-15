#!/bin/bash

set -e

# # #

tag="20$(date "+%2y%2m%2d")"
CHORE_PATH="$(realpath $(dirname $0))"

# build develop image
REAL_DOCKER_FILE="${CHORE_PATH}/develop.dock"
echo "[*] Building 'hzztai/develop:${tag}' by: '${REAL_DOCKER_FILE}'"
docker buildx build \
  --build-context \
  sdk=/home/wangqiang/workspace/lamia/sdk \
  -t "hzztai/develop:${tag}" \
  -f "${REAL_DOCKER_FILE}" \
  "${@:2}" .

echo "[*] Build 'hzztai/develop:${tag}' done."
