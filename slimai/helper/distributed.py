from typing import Any
import threading
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
  FullyShardedDataParallel as FSDP, 
  MixedPrecision, ShardingStrategy, StateDictType, FullStateDictConfig
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy, size_based_auto_wrap_policy, transformer_auto_wrap_policy
from .utils import dist_env, PytorchNetworkUtils
from .structure import DataSample


__all__ = ["Distributed", "FSDPLayerWapper"]


class Distributed(object):
  """A class to handle model distributed parallel.
  
  Attributes:
    parallel_mode (str): The parallel mode to use.
    mix_precision (str): The mixed precision to use.
    env (dist_env): The environment to use pytorch distributed.
  """
  _instance = None
  _lock = threading.Lock()

  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      with cls._lock:
        if cls._instance is None:
          cls._instance = super().__new__(cls)
          cls._instance._initialized = False
    return cls._instance

  def __init__(self, 
               parallel_mode="ddp", 
               default_dtype="fp32", 
               mix_precision="bf16", 
    ):
    if self._initialized:
      return

    assert (
      parallel_mode in ["auto", "ddp", "fsdp"]
    ), "parallel_mode must be 'auto' or 'ddp' or 'fsdp', but got {}".format(parallel_mode)
    if parallel_mode == "auto":
      parallel_mode = "fsdp"

    if self.env.global_world_size == 1:
      parallel_mode = "ddp"
    # TODO: currently, fsdp is not supported, use ddp instead
    parallel_mode = "ddp"
    
    self.parallel_mode = parallel_mode

    assert (
      default_dtype in ["fp16", "bf16", "fp32"]
    ), "default_dtype must be 'fp16', 'bf16', or 'fp32', but got {}".format(default_dtype)
    self.default_dtype = default_dtype
    torch.set_default_dtype(self.dtype)

    assert (
      mix_precision in ["fp16", "bf16", "fp32"]
    ), "mix_precision must be 'fp16', 'bf16', or 'fp32', but got {}".format(mix_precision)
    self.mix_precision = mix_precision

    self._initialized = True
    return
  
  @property
  def dtype(self):
    return dict(
      fp16=torch.float16,
      bf16=torch.bfloat16,
      fp32=torch.float32
    )[self.default_dtype]

  @property
  def mix_dtype(self):
    return dict(
      fp16=torch.float16,
      bf16=torch.bfloat16,
      fp32=torch.float32
    )[self.mix_precision]
  
  def __repr__(self):
    return f"Distributed(parallel_mode={self.parallel_mode}, mix_precision={self.mix_precision})"
  __str__ = __repr__

  @classmethod
  def create(cls, *args, **kwargs):
    return cls(*args, **kwargs)

  @property
  def env(self):
    return dist_env
  
  def prepare_for_distributed(self, *args) -> Any:
    """
    Prepare multiple components for distributed training support.
    
    Args:
      *args: The components to prepare.
      
    Returns:
      The prepared components.
    """
    prepared = [self._prepare_component(component) for component in args]
    if len(prepared) == 1:
      prepared = prepared[0]
    return prepared
  
  def _prepare_component(self, component):
    """
    Prepare a component by type for distributed training support.
    
    Args:
      component: The component to prepare.
      
    Returns:
      The prepared component.
    """
    if isinstance(component, torch.optim.Optimizer):
      component.load_state_dict(component.state_dict()) # cast optimizer to proper device
      return component
    elif isinstance(component, torch.optim.lr_scheduler.LRScheduler):
      component.load_state_dict(component.state_dict()) # cast scheduler to proper device
      return component
    elif isinstance(component, torch.nn.ModuleDict):
      component = torch.nn.ModuleDict({
        k: self._wrap_module(m)
        for (k, m) in component.items()
      })
    elif isinstance(component, torch.nn.Module):
      component = self._wrap_module(component)
    elif isinstance(component, torch.Tensor):
      component = component.to(self.env.device_module.current_device())
    elif isinstance(component, dict):
      component = {
        k: self._prepare_component(v)
        for (k, v) in component.items()
      }

    return component
  
  def _wrap_module(self, module):
    """
    Wrap a PyTorch module with distributed training support.
    
    Args:
      module: torch.nn.Module to be wrapped
      
    Returns:
      Wrapped module with DDP or FSDP based on parallel_mode
    """
    module = module.to(f"{self.env.device_type}:{self.env.local_rank}")
    if not self.env.is_dist_initialized():
      return module
    
    module_size = PytorchNetworkUtils.get_params_size(
      module, grad_mode="trainable", magnitude="digit")
    
    if module_size == 0:
      return module
    
    if self.parallel_mode == "ddp":
      # go with DDP
      return DDP(
        module, static_graph=True
      )
    elif self.parallel_mode == "fsdp":
      # go with FSDP

      # Configure FSDP with mixed precision for better performance
      mixed_precision_policy = MixedPrecision(
        param_dtype=self.mix_dtype,
        reduce_dtype=self.mix_dtype,
        buffer_dtype=self.mix_dtype,
      )

      wrapped_module = FSDPLayerWapper(module=module)
      wrapped_policy = ModuleWrapPolicy(
        module_classes=[FSDPLayerWapper]
      )
      
      module = torch.compile(module)
      module = FSDP(
        wrapped_module,
        device_id=self.env.device_module.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision_policy,
        auto_wrap_policy=wrapped_policy,
        sync_module_states=True,
        use_orig_params=True,
      )
      return module
    
    # go with original module when parallel mode is not recognized
    return module

  def get_summon_module(self, module):
    """
    Get the fully unwrapped module for the given Module or ModuleDict.
    """
    def _to_uncurated_module(module):
      if self.parallel_mode == "ddp":
        return getattr(module, "module")
      elif self.parallel_mode == "fsdp":
        return getattr(module, "module")
      else:
        return module

    if isinstance(module, torch.nn.ModuleDict):
      module = torch.nn.ModuleDict({
        k: _to_uncurated_module(m)
        for (k, m) in module.items()
      })
    elif isinstance(module, torch.nn.Module):
      module = _to_uncurated_module(module)

    return module

  def copy_cpu_offload_state_dict(self, module):
    state_dict = module.state_dict()
    return DataSample(**state_dict).cpu().to_dict()


##### FSDP Layer Wapper #####
class FSDPLayerWapper(torch.nn.Module):
  def __init__(self, *, module=None):
    assert (
      module is not None
    ), "module must be provided"
    super().__init__()
    self.module = module
    return

  def forward(self, *args, **kwargs):
    return self.module(*args, **kwargs)

  def __repr__(self):
    return f"{self.__class__.__name__}(module={self.module})"

  def __str__(self):
    return self.__repr__()
