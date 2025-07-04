import os
import itertools
import datetime
import torch
import numpy as np
import random
from typing import Dict
import torch.distributed as dist


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
    self.accelerator = "cpu"
    self.timeout = datetime.timedelta(seconds=60)
    accelerator_candidates = ["cpu", "cuda", "mps", "mkldnn", "xla", "npu"]
    self.supported_accelerators = []
    for accelerator in accelerator_candidates:
      self.try_register_accelerator(accelerator)
    return
  
  def set_seed(self, *, seed=10482):
    deterministic = False if seed is None else True

    torch.use_deterministic_algorithms(deterministic, warn_only=False)

    if deterministic:
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      np.random.seed(seed)
      random.seed(seed)
    
    if self.accelerator == "cuda":
      torch.backends.cudnn.benchmark = (not deterministic)
      torch.backends.cudnn.deterministic = deterministic

      if deterministic:
        os.putenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
      else:
        os.unsetenv("CUBLAS_WORKSPACE_CONFIG")
    
    return

  def is_main_process(self, local=True):
    # Check if the current process is the main process (local or global)
    rank = self.local_rank if local else self.global_rank
    return rank == 0

  def is_dist_initialized(self):
    # Check if the distributed environment is initialized
    return dist.is_initialized()

  @classmethod
  def get_free_port(cls):
    import socket
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port
  
  @property
  def device(self):
    return torch.device(self.accelerator)
  
  @property
  def device_module(self):
    return getattr(torch, self.accelerator)

  def try_register_accelerator(self, accelerator, rebase=False):
    if accelerator is None:
      return
    if accelerator not in self.supported_accelerators:
      device = getattr(torch, accelerator, None)
      if device is not None and device.is_available():
        self.supported_accelerators.append(accelerator)
    if rebase:
      if accelerator in self.supported_accelerators:
        self.accelerator = accelerator
      else:
        raise ValueError(f"Accelerator {accelerator} is not supported")
    return

  def init_dist(self, *, device=None, timeout=None, seed=10482):
    """Initialize distributed environment."""

    # check if device is supported and rebase the device as default device
    self.try_register_accelerator(device, rebase=True)

    backend = dict(
      cpu="gloo",
      cuda="nccl",
      npu="hccl",
      mps="gloo",
      mkldnn="gloo",
      xla="gloo",
    ).get(self.accelerator, None)

    if timeout is not None:
      self.timeout = datetime.timedelta(seconds=timeout)

    self.set_seed(seed=seed)

    # initialize distributed environment
    if not dist.is_initialized():
      if self.env.get("WORLD_SIZE", None) is None: # in non ddp mode, MASTER_ADDR and MASTER_PORT are not set, mannually set them to use distributed training
        self.env["MASTER_ADDR"] = os.environ["MASTER_ADDR"] = "localhost"
        self.env["MASTER_PORT"] = os.environ["MASTER_PORT"] = "12345"
      
      self.device_module.set_device(self.local_rank)
      dist.init_process_group(
        backend=backend, 
        rank=self.global_rank, 
        world_size=self.global_world_size,
        timeout=self.timeout, 
      )

    return
  
  def broadcast(self, data, from_rank=0):
    """Broadcast data to all processes."""
    if not self.is_dist_initialized():
      return data
    
    output = [data] # wrap to list to use broadcast_object_list
    dist.broadcast_object_list(output, src=from_rank) # auto barrier across all processes
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

    if work := dist.barrier(async_op=True):
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
    output = list(itertools.chain(*output)) # type: ignore
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
    return "Distributed {}, LOCAL RANK: {} of {}-th NODE, GLOBAL RANK: {} in all {} NODES".format(
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
