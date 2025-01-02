import os
import itertools
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class DistEnv(object):
  env = {
    k: os.environ.get(k, None) for k in [
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
    ]}

  def is_main_process(self):
    # Check if the current process is the main process
    return self.local_rank == 0

  def is_dist_initialized(self):
    # Check if the distributed environment is initialized
    return dist.is_initialized()
  
  def init_dist(self, module=None, backend="nccl"):
    """Initialize distributed environment."""
    if not dist.is_initialized():
      dist.init_process_group(backend=backend)
      torch.cuda.set_device(self.local_rank)
      torch.backends.cudnn.benchmark = True

    if module is not None:
      module.to(self.local_rank)
      module = DDP(module, device_ids=[self.local_rank])
    return module
  
  def sync(self, data=None, tensor_op=dist.ReduceOp.AVG):
    """Reduce data (Tensor or Dict of Tensor) across all processes."""
    
    def _sync_all_types(_data):
      if isinstance(_data, torch.Tensor):
        dist.all_reduce(_data, op=tensor_op)
        return _data
      elif isinstance(_data, dict):
        return {k: _sync_all_types(v) for k, v in _data.items()}
      else:
        raise ValueError(f"Unsupported data type: {type(_data)}")

    if data is not None:
      data = _sync_all_types(data)

    dist.barrier()
    return data

  def collect(self, data):
    """Collect list of objects from all processes and merge into a single list."""
    assert (
      isinstance(data, list)
    ), "collect data must be a list, but got {}".format(type(data))

    output = [None for _ in range(self.global_world_size)]
    dist.all_gather_object(output, data)
    dist.barrier()
    output = list(itertools.chain(*output))
    return output

  def close_dist(self):
    """Close the distributed environment."""
    if dist.is_initialized():
      dist.destroy_process_group()
    return
  
  @property
  def desc(self):
    """Describe the distributed environment."""
    return "LOCAL RANK: {} of {}-th NODE, GLOBAL RANK: {} in all {} NODES".format(
      self.local_rank, self.local_world_size, self.global_rank, self.global_world_size
    )
  
  @property
  def local_rank(self):
    return int(self.env["LOCAL_RANK"])
  
  @property
  def global_rank(self):
    return int(self.env["RANK"])
  
  @property
  def local_world_size(self):
    return int(self.env["LOCAL_WORLD_SIZE"])
  
  @property
  def global_world_size(self):
    return int(self.env["WORLD_SIZE"])
  
  @property
  def master_addr(self):
    return self.env["MASTER_ADDR"]

  @property
  def master_port(self):
    return int(self.env["MASTER_PORT"])

  @property
  def torchelastic_restart_count(self):
    return int(self.env["TORCHELASTIC_RESTART_COUNT"])

  @property
  def torchelastic_max_restarts(self):
    return int(self.env["TORCHELASTIC_MAX_RESTARTS"])

  @property
  def torchelastic_run_id(self):
    return self.env["TORCHELASTIC_RUN_ID"]
  

dist_env = DistEnv()
