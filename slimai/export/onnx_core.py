from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import onnx
import torch

try:
  from onnxsim import simplify
except Exception:
  def simplify(model):  # type: ignore
    return model, True


def export_onnx(
  module: torch.nn.Module,
  dummy_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
  onnx_path: Path,
  *,
  input_names: Sequence[str],
  output_names: Sequence[str],
  dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
  opset_version: int = 17,
  device: str = "cpu",
) -> Path:
  module = module.to(device).eval()
  if isinstance(dummy_inputs, torch.Tensor):
    dummy_tuple = (dummy_inputs.to(device),)
  else:
    dummy_tuple = tuple(tensor.to(device) for tensor in dummy_inputs)

  onnx_path = Path(onnx_path)
  onnx_path.parent.mkdir(parents=True, exist_ok=True)

  export_kwargs = dict(
    export_params=True,
    opset_version=opset_version,
    do_constant_folding=True,
    input_names=list(input_names),
    output_names=list(output_names),
    dynamic_axes=dynamic_axes,
  )
  with torch.inference_mode():
    try:
      torch.onnx.export(
        module,
        dummy_tuple if len(dummy_tuple) > 1 else dummy_tuple[0],
        str(onnx_path),
        dynamo=False,
        **export_kwargs,
      )
    except TypeError:
      torch.onnx.export(
        module,
        dummy_tuple if len(dummy_tuple) > 1 else dummy_tuple[0],
        str(onnx_path),
        **export_kwargs,
      )

  onnx_model = onnx.load(str(onnx_path))
  onnx.checker.check_model(onnx_model, full_check=True)
  simplified_model, check = simplify(onnx_model)
  if check:
    onnx.save_model(simplified_model, str(onnx_path))
  return onnx_path
