import os
import itertools
import datetime
import torch
import numpy as np
import random
from typing import Dict
import torch.distributed as dist
from . import singleton


__all__ = [
  "get_dist_env",
]

def get_dist_env():
  return DistEnv()


@singleton.singleton_wrapper
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

  accelerator = "cpu"
  timeout = datetime.timedelta(seconds=60)
  supported_accelerators = []

  ########################################

  def __init__(self) -> None:
    accelerator_candidates = ["cpu", "cuda", "mps", "mkldnn", "xla", "npu"]
    for accelerator in accelerator_candidates:
      self.try_register_accelerator(accelerator)
    return
  
  @classmethod
  def set_seed(cls, *, seed=10482):
    deterministic = False if seed is None else True

    torch.use_deterministic_algorithms(deterministic, warn_only=False)

    if deterministic:
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      np.random.seed(seed)
      random.seed(seed)
    
    if cls.accelerator == "cuda":
      torch.backends.cudnn.benchmark = (not deterministic)
      torch.backends.cudnn.deterministic = deterministic

      if deterministic:
        os.putenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
      else:
        os.unsetenv("CUBLAS_WORKSPACE_CONFIG")
    
    return

  @classmethod
  def is_main_process(cls, local=True):
    # Check if the current process is the main process (local or global)
    rank = cls.local_rank if local else cls.global_rank
    return rank == 0

  @classmethod
  def is_dist_initialized(cls):
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
  
  @singleton.classproperty
  def device(cls):
    return torch.device(cls.accelerator)
  
  @singleton.classproperty
  def device_module(cls):
    return getattr(torch, cls.accelerator)

  @classmethod
  def try_register_accelerator(cls, accelerator, rebase=False):
    if accelerator is None:
      return
    if accelerator not in cls.supported_accelerators:
      device = getattr(torch, accelerator, None)
      if device is not None and device.is_available():
        cls.supported_accelerators.append(accelerator)
    if rebase:
      if accelerator in cls.supported_accelerators:
        cls.accelerator = accelerator
      else:
        raise ValueError(f"Accelerator {accelerator} is not supported")
    return

  @classmethod
  def init_dist(cls, *, device=None, timeout=None, seed=10482):
    """Initialize distributed environment."""

    # check if device is supported and rebase the device as default device
    cls.try_register_accelerator(device, rebase=True)

    backend = dict(
      cpu="gloo",
      cuda="nccl",
      npu="hccl",
      mps="gloo",
      mkldnn="gloo",
      xla="gloo",
    ).get(cls.accelerator, None)

    if timeout is not None:
      cls.timeout = datetime.timedelta(seconds=timeout)

    cls.set_seed(seed=seed)

    # initialize distributed environment
    if not dist.is_initialized():
      if cls.env.get("WORLD_SIZE", None) is None: # in non ddp mode, MASTER_ADDR and MASTER_PORT are not set, mannually set them to use distributed training
        cls.env["MASTER_ADDR"] = os.environ["MASTER_ADDR"] = "localhost"
        cls.env["MASTER_PORT"] = os.environ["MASTER_PORT"] = "12345"
      
      cls.set_start_method("fork")
      cls.device_module.set_device(cls.local_rank)
      dist.init_process_group(
        backend=backend, 
        rank=cls.global_rank, 
        world_size=cls.global_world_size,
        timeout=cls.timeout, 
      )
    return

  @classmethod
  def set_start_method(cls, method="fork"):
    if torch.multiprocessing.get_start_method() != method:
      torch.multiprocessing.set_start_method(method, force=True)
    return
  
  @classmethod
  def broadcast(cls, data, from_rank=0):
    """Broadcast data to all processes."""
    if not cls.is_dist_initialized():
      return data
    
    output = [data] # wrap to list to use broadcast_object_list
    dist.broadcast_object_list(output, src=from_rank) # auto barrier across all processes
    data = output[0]
    return data
  
  @classmethod
  def sync(cls, data=None, tensor_op=dist.ReduceOp.AVG):
    """Reduce data (Tensor or Dict of Tensor) across all processes."""
    if not cls.is_dist_initialized():
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
      work.wait(timeout=cls.timeout)
    return data

  @classmethod
  def collect(cls, data):
    """Collect list of objects from all processes and merge into a single list.
    This collect may need sea of memory so that lead into crash.
    """
    if not cls.is_dist_initialized():
      return data
    
    assert (
      isinstance(data, list)
    ), "collect data must be a list, but got {}".format(type(data))

    output = [None for _ in range(cls.global_world_size)]
    dist.all_gather_object(output, data) # auto barrier across all processes
    output = list(itertools.chain(*output)) # type: ignore
    return output

  @classmethod
  def close_dist(cls):
    """Close the distributed environment."""
    cls.sync()
    if dist.is_initialized():
      dist.destroy_process_group()
    return
  
  @singleton.classproperty
  def desc(cls):
    """Describe the distributed environment."""
    return "Accelerator: {}, Distributed {}, LOCAL RANK: {} of {}-th NODE, GLOBAL RANK: {} in all {} NODES".format(
      cls.accelerator,
      "enabled" if cls.is_dist_initialized() else "disabled",
      cls.local_rank, cls.local_world_size, cls.global_rank, cls.global_world_size
    )
  
  @singleton.classproperty
  def local_rank(cls):
    return int(cls.env.get("LOCAL_RANK", 0))
  
  @singleton.classproperty
  def global_rank(cls):
    return int(cls.env.get("RANK", 0))
  
  @singleton.classproperty
  def local_world_size(cls):
    return int(cls.env.get("LOCAL_WORLD_SIZE", 1))
  
  @singleton.classproperty
  def global_world_size(cls):
    return int(cls.env.get("WORLD_SIZE", 1))
  
  @singleton.classproperty
  def master_addr(cls):
    return cls.env.get("MASTER_ADDR", "localhost")

  @singleton.classproperty
  def master_port(cls):
    return int(cls.env.get("MASTER_PORT", str(cls.get_free_port())))

  @singleton.classproperty
  def torchelastic_restart_count(cls):
    return int(cls.env.get("TORCHELASTIC_RESTART_COUNT", "0"))

  @singleton.classproperty
  def torchelastic_max_restarts(cls):
    return int(cls.env.get("TORCHELASTIC_MAX_RESTARTS", "0"))

  @singleton.classproperty
  def torchelastic_run_id(cls):
    return cls.env.get("TORCHELASTIC_RUN_ID", "0")
