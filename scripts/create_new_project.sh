#!/bin/bash

set -e

# # #
PROJECT_NAME="$1"

if [ -z "${PROJECT_NAME}" ]
then
  echo "PROJECT_NAME cannot be empty!"
  echo "Usage: bash $(readlink -f $0) PROJECT_NAME"
  exit 1
fi

WORKING_PATH="$(git rev-parse --show-toplevel)"

CREATE_DIR="${WORKING_PATH}/projects"

echo "[*] Create Project: ${PROJECT_NAME}"
echo "[*] Under: ${CREATE_DIR}"

# # #
PROJECT_DIR="${CREATE_DIR}/${PROJECT_NAME}"

echo "[*] Setting up "
for name in configs docs jupyter temp; do \
  subfolder="${PROJECT_DIR}/${name}"; \
  echo "Create Folder: ${subfolder}"; \
  mkdir -p "${subfolder}"; \
done

for name in app archive data ckpt tags yml; do \
  subfolder="${PROJECT_DIR}/${name}"; \
  echo "Create Folder: ${subfolder}"; \
  mkdir -p "${subfolder}"; \
done

exit 0
