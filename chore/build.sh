#!/bin/bash

set -e

# # #

cd "$(dirname $0)"

CHORE_PATH="$(cd "$(dirname "$0")" && pwd)"
REAL_DOCKER_FILE="${CHORE_PATH}/Dockerfile"

tag="20$(date "+%2y%2m%2d")"
echo "[*] Building 'hzztai/develop:${tag}' by: '${REAL_DOCKER_FILE}'"

docker buildx build \
  --build-context \
  sdk=/home/wangqiang/workspace/projects/sdk/reader_archives/latest/sdk \
  -t "hzztai/develop:${tag}" \
  -f "${REAL_DOCKER_FILE}" \
  "${@:2}" .

echo "[*] Build 'hzztai/develop:${tag}' done."
