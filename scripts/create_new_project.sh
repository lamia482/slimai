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

WORKING_PATH="/hzztai"

CREATE_DIR="${WORKING_PATH}/projects"

echo "[*] Create Project: ${PROJECT_NAME}"
echo "[*] Under: ${CREATE_DIR}"

# show project path, and require user to confirm
echo "[*] Project Path: ${CREATE_DIR}/${PROJECT_NAME}"
read -p "Are you sure to create this project? (y/n): " confirm
if [ "${confirm,,}" != "y" ] && [ "${confirm,,}" != "Y" ]; then
  echo "[-] Abort"
  exit 1
fi

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
