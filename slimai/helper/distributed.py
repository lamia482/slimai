import functools
import threading
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
  FullyShardedDataParallel as FSDP, 
  MixedPrecision, ShardingStrategy, StateDictType, FullStateDictConfig
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from .utils import dist_env, PytorchNetworkUtils


__all__ = ["Distributed"]


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

  def __init__(self, parallel_mode="ddp", mix_precision="bf16"):
    if self._initialized:
      return

    assert (
      parallel_mode in ["auto", "ddp"]
    ), "parallel_mode must be 'auto' or 'ddp', but got {}".format(parallel_mode)
    self.parallel_mode = "ddp"

    assert (
      mix_precision in ["fp16", "bf16", "fp32"]
    ), "mix_precision must be 'fp16', 'bf16', or 'fp32', but got {}".format(mix_precision)
    self.mix_precision = mix_precision

    self._initialized = True
    return
  
  @property
  def mix_dtype(self):
    return dict(
      fp16=torch.float16,
      bf16=torch.bfloat16,
      fp32=torch.float32
    )[self.mix_precision]
  
  def __repr__(self):
    return f"Distributed(parallel_mode={self.parallel_mode}, mix_precision={self.mix_precision})\n{self.env.desc}"
  __str__ = __repr__

  @classmethod
  def create(cls, *args, **kwargs):
    return cls(*args, **kwargs)

  @property
  def env(self):
    return dist_env
  
  def prepare_for_distributed(self, *args):
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
    
    if self.parallel_mode == "ddp":
      return module if not PytorchNetworkUtils.get_params_size(
        module, grad_mode="trainable", magnitude="digit") else DDP(
        module, static_graph=True
      )
    elif self.parallel_mode == "fsdp":
      # Configure FSDP with mixed precision for better performance
      my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=20000
      )
      mixed_precision_policy = MixedPrecision(
        param_dtype=self.mix_dtype,
        reduce_dtype=self.mix_dtype,
        buffer_dtype=self.mix_dtype,
      )
      
      return FSDP(
        module,
        device_id=self.env.local_rank,
        auto_wrap_policy=my_auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
      )
    return module

  def get_summon_module(self, module):
    """
    Get the fully unwrapped module for the given Module or ModuleDict.
    """
    def _to_uncurated_module(module):
      if self.parallel_mode == "ddp":
        return getattr(module, "module", module)
      elif self.parallel_mode == "fsdp":
        with module.summon_full_params():
          return module.module
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

    if isinstance(module, torch.nn.Module):  
      for key, value in state_dict.items():
        state_dict[key] = value.cpu()

    return state_dict
    
