import torch
from typing import Optional
from .help_utils import print_log
from .utils.network import PytorchNetworkUtils
from .distributed import Distributed


__all__ = ["Gradient"]


class Gradient(object):
  """A class to handle gradient operations during training.
  
  This class provides functionality for gradient accumulation, scaling, and clipping.
  
  Attributes:
    amp (bool): Whether to use automatic mixed precision
    accumulation_every_n_steps (int): Number of steps to accumulate gradients
    clip (Optional[float]): Gradient clipping value
    scaler (torch.amp.GradScaler): Gradient scaler for mixed precision
  """
  
  def __init__(
    self,
    amp: bool = False,
    accumulation_every_n_steps: int = 1,
    clip: Optional[float] = None
  ):
    """Initialize Gradient handler.
    
    Args:
      amp: Whether to use automatic mixed precision (default: False)
      accumulation_every_n_steps: Number of steps to accumulate gradients (default: 1)
      clip: Gradient clipping value (default: None)
    """
    self.dist = Distributed.create()

    # Validate inputs
    assert (
      accumulation_every_n_steps >= 1
    ), "accumulation_every_n_steps must be greater than or equal to 1"
    
    # Set up gradient configuration
    self.amp = amp and torch.cuda.is_bf16_supported()
    self.accumulation_every_n_steps = accumulation_every_n_steps
    self.clip = clip
    
    # Initialize gradient scaler for mixed precision
    if self.dist.parallel_mode == "fsdp":
      from torch.distributed.fsdp import sharded_grad_scaler
      GradScaler = sharded_grad_scaler.ShardedGradScaler
    else:
      GradScaler = torch.amp.GradScaler
    
    self.scaler = GradScaler(self.dist.env.device_type, enabled=self.amp)
    
    # Handle distributed training
    if (not self.dist.env.is_dist_initialized()) and (self.accumulation_every_n_steps > 1):
      self.accumulation_every_n_steps = 1
      print_log(f"gradient accumulation is not supported yet in non-dist mode, set to 1", level="WARNING")
    return
  
  def step(
    self,
    model: torch.nn.Module,
    solver: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss: torch.Tensor,
    i_step: int,
    total_steps: int,
    accumulation_i_step: int
  ) -> None:
    """Perform a single gradient step with accumulation.
    
    Args:
      model: PyTorch model
      solver: Optimizer
      scheduler: Learning rate scheduler
      loss: Loss tensor
      i_step: Current step index
      total_steps: Total number of steps
      accumulation_i_step: Current accumulation step index
    """
    if i_step == 0:  # clear grad before first step
      solver.zero_grad()

    # Scale loss with AMP mode and backward
    self.scaler.scale(loss / self.accumulation_every_n_steps).backward()
    
    # Handle gradient clipping
    if self.clip is not None and self.scaler.is_enabled():
      self.scaler.unscale_(solver)
    PytorchNetworkUtils.clip_gradients(model, self.clip)

    # Step optimizer and update learning rate after gradient accumulation
    if (self.accumulation_every_n_steps == 1  # no grad accumulation mode
        ) or (i_step == total_steps - 1  # last step to accumulate grad
        ) or (accumulation_i_step == self.accumulation_every_n_steps - 1  # meet accumulation steps
        ):
      self.scaler.step(solver)
      self.scaler.update()
      solver.zero_grad()
      scheduler.step()
      
    else:
      self.scaler.scale(loss / self.accumulation_every_n_steps).backward()

    return
