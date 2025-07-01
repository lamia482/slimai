import onnx
try:
  from onnxsim import simplify
except:
  simplify = lambda _: (_, True)
import torch
from pathlib import Path
from functools import partial
from mmengine.model.utils import revert_sync_batchnorm
from slimai.helper import help_build, help_utils
from slimai.helper.utils import PytorchNetworkUtils
from slimai.helper.distributed import Distributed
from slimai.models.component.pipeline import Pipeline
from slimai.models.arch.base_arch import BaseArch


class Exporter(torch.nn.Module):
  """A class for exporting PyTorch models to ONNX format."""

  def __init__(self, ckpt_path, *, disable_log=False):
    """Initialize exporter with checkpoint path."""
    super().__init__()
    self.ckpt_path = ckpt_path
    self.disable_log = disable_log
    
    help_utils.print_log(f"Loading checkpoint from {ckpt_path}", disable_log=self.disable_log)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    assert (
      {"model", "weight"}.issubset(set(ckpt.keys()))
    ), "Invalid checkpoint keys, expect 'model' and 'weight', but got {}".format(ckpt.keys())

    dist = Distributed.create()

    # turn to non-ddp mode
    help_utils.print_log("Building model architecture", disable_log=self.disable_log)
    arch: BaseArch = help_build.build_model(ckpt["model"]) # type: ignore
    #TODO: fit fsdp
    arch.model = dist.get_summon_module(arch.model)
    arch.load_state_dict(ckpt["weight"], strict=True)
    model = dist.get_summon_module(arch.export_model())
    model = revert_sync_batchnorm(model)

    self.model = model.eval()
    help_utils.print_log("Model initialized successfully", disable_log=self.disable_log)
    help_utils.print_log(f"Model parameters: {PytorchNetworkUtils.get_params_size(self.model)}", disable_log=self.disable_log)
    return

  @property
  def device(self):
    """Get device of the model."""
    return next(self.model.parameters()).device
  
  def forward(self, input_tensor):
    """Forward function of the model."""
    forward_func = self.model.forward
    if isinstance(self.model, Pipeline):
      forward_func = partial(forward_func, return_flow=True)
    return forward_func(input_tensor)
  
  def export(self, output_dir, *, format="onnx"):    
    """Export model to ONNX format."""
    help_utils.print_log(f"Exporting model to {format} format", disable_log=self.disable_log)
    help_utils.print_log(f"Using device: {self.device}", disable_log=self.disable_log)
    input_tensor = torch.randn(1, 3, 224, 224).to(self.device)
    
    help_utils.print_log("Running model inference", disable_log=self.disable_log)
    with torch.inference_mode():
      pytorch_output = self(input_tensor)
    help_utils.print_log(f"Output shape: {pytorch_output.shape if isinstance(pytorch_output, torch.Tensor) else {k: v.shape for k,v in pytorch_output.items()}}", disable_log=self.disable_log)

    nested_model_file = Path(output_dir) / "-".join(Path(self.ckpt_path).with_suffix(f".{format}").parts[-3:])
    nested_model_file.parent.mkdir(parents=True, exist_ok=True)

    help_utils.print_log(f"Exporting model to {nested_model_file}", disable_log=self.disable_log)
    if format == "onnx":
      nested_output = self._export_onnx(input_tensor, nested_model_file)
    else:
      raise ValueError("Invalid format, expect 'onnx', but got {}".format(format))
        
    assert ( # TODO: validate output
      True # pytorch_output == nested_output
    ), "Output mismatch"

    help_utils.print_log("Model exported successfully", disable_log=self.disable_log)
    return

  ############################################################
  # ONNX export
  ############################################################
  def _export_onnx(self, input_tensor, onnx_file):
    """Export model to ONNX format."""
    export_options = dict(
      export_params=True, 
      opset_version=17, 
      do_constant_folding=True, 
      input_names=["input"], 
      output_names=["output"], 
      dynamic_axes={
        "input": {0: "batch_size", 1: "channel", 2: "height", 3: "width"}, 
        "output": {0: "batch_size"}, 
      }
    )
    torch.onnx.export(self, input_tensor, onnx_file, **export_options) # type: ignore
    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model, full_check=True)
    onnx_model, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save_model(onnx_model, onnx_file)
    return 