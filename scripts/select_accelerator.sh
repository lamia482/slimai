#!/bin/bash

_select_accelerator_user_device() {
  local args=("$@")
  local i arg
  local user_device=""

  for ((i = 0; i < ${#args[@]}; i++)); do
    arg="${args[$i]}"
    if [ "${arg}" = "--device" ] && [ $((i + 1)) -lt ${#args[@]} ]; then
      user_device="${args[$((i + 1))]}"
    elif [[ "${arg}" == --device=* ]]; then
      user_device="${arg#--device=}"
    fi
  done

  echo "${user_device}"
}

_select_accelerator_auto_device() {
  if [ -n "${ASCEND_VISIBLE_DEVICES}" ]; then
    echo "npu"
    return 0
  elif [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    echo "cuda"
    return 0
  fi

  python - <<'PY'
import torch

try:
  import torch_npu  # type: ignore # noqa: F401
except Exception:
  torch_npu = None

if torch_npu is not None and hasattr(torch, "npu") and torch.npu.is_available():
  print("npu")
elif torch.cuda.is_available():
  print("cuda")
else:
  print("cpu")
PY
}

_select_accelerator_visible_device_count() {
  local visible_devices="$1"
  local normalized
  local count=0
  local device_id

  if [ -z "${visible_devices}" ]; then
    echo ""
    return 0
  fi

  normalized="${visible_devices//[[:space:]]/}"
  if [ -z "${normalized}" ]; then
    echo "0"
    return 0
  fi

  if [ "${normalized}" = "-1" ] || [ "${normalized}" = "none" ] || \
    [ "${normalized}" = "None" ] || [ "${normalized}" = "void" ] || \
    [ "${normalized}" = "Void" ] || [ "${normalized}" = "NoDevFiles" ]; then
    echo "0"
    return 0
  fi

  IFS=',' read -ra visible_device_ids <<< "${normalized}"
  for device_id in "${visible_device_ids[@]}"; do
    if [ -n "${device_id}" ]; then
      count=$((count + 1))
    fi
  done

  echo "${count}"
}

_select_accelerator_device_count() {
  local device="$1"
  local visible_count

  if [ "${device}" = "cuda" ]; then
    visible_count="$(_select_accelerator_visible_device_count "${CUDA_VISIBLE_DEVICES}")"
    if [ -n "${visible_count}" ]; then
      echo "${visible_count}"
      return 0
    fi

    python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
  elif [ "${device}" = "npu" ]; then
    visible_count="$(_select_accelerator_visible_device_count "${ASCEND_VISIBLE_DEVICES}")"
    if [ -n "${visible_count}" ]; then
      echo "${visible_count}"
      return 0
    fi

    python - <<'PY'
import torch

try:
  import torch_npu  # type: ignore # noqa: F401
except Exception:
  print(0)
else:
  if hasattr(torch, "npu") and torch.npu.is_available():
    print(torch.npu.device_count())
  else:
    print(0)
PY
  elif [ "${device}" = "cpu" ]; then
    echo "1"
  else
    echo "0"
  fi
}

select_accelerator() {
  local user_device
  local device
  local device_count

  user_device="$(_select_accelerator_user_device "$@")"
  device="${user_device:-${DEVICE:-auto}}"

  if [ "${device}" = "auto" ]; then
    device="$(_select_accelerator_auto_device)"
  fi

  if [ "${device}" != "cuda" ] && [ "${device}" != "npu" ] && [ "${device}" != "cpu" ]; then
    echo "Error: unsupported device '${device}', expected one of auto/cuda/npu/cpu" >&2
    return 1
  fi

  device_count="$(_select_accelerator_device_count "${device}")"
  if [[ -z "${device_count}" || "${device_count}" == *[!0-9]* ]]; then
    echo "Error: failed to resolve device count for '${device}', got '${device_count}'" >&2
    return 1
  fi

  if [ "${device}" != "cpu" ] && [ "${device_count}" -le 0 ]; then
    echo "Error: device '${device}' requested but no visible devices were found" >&2
    return 1
  fi

  if [ "${device}" = "cuda" ] && [ -z "${CUDA_VISIBLE_DEVICES}" ] && [ "${device_count}" -gt 0 ]; then
    export CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((device_count - 1)))
  elif [ "${device}" = "npu" ] && [ -z "${ASCEND_VISIBLE_DEVICES}" ] && [ "${device_count}" -gt 0 ]; then
    export ASCEND_VISIBLE_DEVICES
    ASCEND_VISIBLE_DEVICES=$(seq -s, 0 $((device_count - 1)))
  fi

  export SELECT_ACCELERATOR_USER_DEVICE="${user_device}"
  export SELECT_ACCELERATOR_DEVICE="${device}"
  export SELECT_ACCELERATOR_DEVICE_COUNT="${device_count}"
  if [ -z "${user_device}" ]; then
    export SELECT_ACCELERATOR_APPEND_DEVICE=1
  else
    export SELECT_ACCELERATOR_APPEND_DEVICE=0
  fi
  return 0
}
