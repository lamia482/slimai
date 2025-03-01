import os
import itertools
import datetime
import torch
from typing import Dict
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .network import PytorchNetworkUtils


class DistEnv(object):
  env = {
    k: os.environ[k] for k in [
      "LOCAL_RANK",
      "RANK", 
      "GROUP_RANK",
      "ROLE_RANK",
      "LOCAL_WORLD_SIZE",
      "WORLD_SIZE",
      "ROLE_WORLD_SIZE",
      "MASTER_ADDR",
      "MASTER_PORT",
      "TORCHELASTIC_RESTART_COUNT",
      "TORCHELASTIC_MAX_RESTARTS",
      "TORCHELASTIC_RUN_ID",
    ] if k in os.environ}
  
  def __init__(self) -> None:
    self.timeout = 60
    return

  def is_main_process(self):
    # Check if the current process is the main process
    return self.local_rank == 0

  def is_dist_initialized(self):
    # Check if the distributed environment is initialized
    return dist.is_initialized()
  
  def init_dist(self, *, module=None, backend="nccl", timeout=None):
    """Initialize distributed environment."""

    # initialize distributed environment when not initialized and WORLD_SIZE is set
    if (not dist.is_initialized()) and (self.env.get("WORLD_SIZE", None) is not None):
      if timeout is not None:
        self.timeout = datetime.timedelta(seconds=timeout)
      dist.init_process_group(backend=backend, 
                              timeout=self.timeout, 
                              )
      torch.cuda.set_device(self.local_rank)
      torch.backends.cudnn.benchmark = True

    if module is not None:
      assert isinstance(
        module, (torch.nn.ModuleDict, torch.nn.Module)
      ), "module must be a torch.nn.Module, but got {}".format(type(module))

      def update_module(q):
        """move to cuda and wrap with DDP if needed"""
        q = q.cuda().to(self.local_rank)
        if self.is_dist_initialized():
          q = self.update2ddp(q) if (
            PytorchNetworkUtils.get_params_size(q, grad_mode="trainable", magnitude="digit") > 0
          ) else q
        return q

      if isinstance(module, torch.nn.ModuleDict):
        module = torch.nn.ModuleDict({
          k: update_module(m)
          for (k, m) in module.items()
        })
      else:
        module = update_module(module)
    return module

  def update2ddp(self, module):
    """Update module to be DDP"""
    module = DDP(module, static_graph=True)
    return module
  
  def broadcast(self, data):
    """Broadcast data to all processes."""
    if not self.is_dist_initialized():
      return data
    
    output = [data] # wrap to list to use broadcast_object_list
    dist.broadcast_object_list(output, src=0) # auto barrier across all processes
    data = output[0]
    return data
  
  def sync(self, data=None, tensor_op=dist.ReduceOp.AVG):
    """Reduce data (Tensor or Dict of Tensor) across all processes."""
    if not self.is_dist_initialized():
      return data
    
    def _sync_all_types(_data):
      if isinstance(_data, torch.Tensor):
        dist.all_reduce(_data, op=tensor_op)
        return _data
      elif isinstance(_data, Dict):
        return {k: _sync_all_types(v) for k, v in _data.items()}
      else:
        raise ValueError(f"Unsupported data type: {type(_data)}")

    if data is not None:
      data = _sync_all_types(data)

    work = dist.barrier(async_op=True)
    work.wait(timeout=self.timeout)
    return data

  def collect(self, data):
    """Collect list of objects from all processes and merge into a single list.
    This collect may need sea of memory so that lead into crash.
    """
    if not self.is_dist_initialized():
      return data
    
    assert (
      isinstance(data, list)
    ), "collect data must be a list, but got {}".format(type(data))

    output = [None for _ in range(self.global_world_size)]
    dist.all_gather_object(output, data) # auto barrier across all processes
    output = list(itertools.chain(*output))
    return output

  def close_dist(self):
    """Close the distributed environment."""
    self.sync()
    if dist.is_initialized():
      dist.destroy_process_group()
    return
  
  @property
  def desc(self):
    """Describe the distributed environment."""
    return "DDP {}, LOCAL RANK: {} of {}-th NODE, GLOBAL RANK: {} in all {} NODES".format(
      "enabled" if self.is_dist_initialized() else "disabled",
      self.local_rank, self.local_world_size, self.global_rank, self.global_world_size
    )
  
  @property
  def local_rank(self):
    return int(self.env.get("LOCAL_RANK", 0))
  
  @property
  def global_rank(self):
    return int(self.env.get("RANK", 0))
  
  @property
  def local_world_size(self):
    return int(self.env.get("LOCAL_WORLD_SIZE", 1))
  
  @property
  def global_world_size(self):
    return int(self.env.get("WORLD_SIZE", 1))
  
  @property
  def master_addr(self):
    return self.env.get("MASTER_ADDR", "localhost")

  @property
  def master_port(self):
    return int(self.env.get("MASTER_PORT", "12345"))

  @property
  def torchelastic_restart_count(self):
    return int(self.env.get("TORCHELASTIC_RESTART_COUNT", "0"))

  @property
  def torchelastic_max_restarts(self):
    return int(self.env.get("TORCHELASTIC_MAX_RESTARTS", "0"))

  @property
  def torchelastic_run_id(self):
    return self.env.get("TORCHELASTIC_RUN_ID", "0")
  

dist_env = DistEnv()
