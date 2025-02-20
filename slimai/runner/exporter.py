import torch
from pathlib import Path
from mmengine.model.utils import revert_sync_batchnorm
from slimai.helper import help_build, help_utils


class Exporter(torch.nn.Module):
  def __init__(self, ckpt_path, *, disable_log=False):
    super().__init__()
    self.ckpt_path = ckpt_path
    self.disable_log = disable_log
    
    help_utils.print_log(f"Loading checkpoint from {ckpt_path}", disable_log=self.disable_log)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    assert (
      {"model", "weight"}.issubset(set(ckpt.keys()))
    ), "Invalid checkpoint keys, expect 'model' and 'weight', but got {}".format(ckpt.keys())

    # turn to non-ddp mode
    help_utils.print_log("Building model architecture", disable_log=self.disable_log)
    arch = help_build.build_model(ckpt["model"])
    arch.model = help_utils.PytorchNetworkUtils.get_module(arch.model)
    arch.load_state_dict(ckpt["weight"], strict=True)
    model = revert_sync_batchnorm(arch.export_model)

    self.model = model.eval()
    help_utils.print_log("Model initialized successfully", disable_log=self.disable_log)
    help_utils.print_log(f"Model parameters: {help_utils.PytorchNetworkUtils.get_params_size(self.model)}", disable_log=self.disable_log)
    return

  @property
  def device(self):
    return next(self.model.parameters()).device
  
  def forward(self, input_tensor):
    return self.model(input_tensor, return_flow=True)
  
  def export(self, output_dir, *, format="onnx"):    
    help_utils.print_log(f"Exporting model to {format} format", disable_log=self.disable_log)
    help_utils.print_log(f"Using device: {self.device}", disable_log=self.disable_log)
    input_tensor = torch.randn(1, 3, 224, 224).to(self.device)
    
    help_utils.print_log("Running model inference", disable_log=self.disable_log)
    with torch.no_grad():
      pytorch_output = self(input_tensor)
    help_utils.print_log(f"Output shape: {pytorch_output.shape if isinstance(pytorch_output, torch.Tensor) else {k: v.shape for k,v in pytorch_output.items()}}", disable_log=self.disable_log)

    nested_model_file = Path(output_dir) / "-".join(Path(self.ckpt_path).with_suffix(f".{format}").parts[-3:])
    nested_model_file.parent.mkdir(parents=True, exist_ok=True)

    help_utils.print_log(f"Saving model to {nested_model_file}", disable_log=self.disable_log)
    if format == "onnx":
      torch.onnx.export(self, input_tensor, nested_model_file)
      help_utils.print_log("Model exported successfully", disable_log=self.disable_log)
    else:
      raise ValueError("Invalid format, expect 'onnx', but got {}".format(format))
    
    return
