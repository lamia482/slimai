import os
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
    return self.local_rank == 0

  def is_dist_initialized(self):
    return dist.is_initialized()
  
  def init_dist(self, module=None, backend="nccl"):
    if not dist.is_initialized():
      dist.init_process_group(backend=backend)
      torch.cuda.set_device(self.local_rank)
      torch.backends.cudnn.benchmark = True

    if module is not None:
      module.to(self.local_rank)
      module = DDP(module, device_ids=[self.local_rank])
    return module
  
  def sync(self, tensor=None, op=dist.ReduceOp.AVG):
    dist.barrier()
    if tensor is not None:
      dist.all_reduce(tensor, op=op)
    return tensor

  def gather(self, data=None):
    dist.barrier()
    if data is not None:
      data = dist.all_gather(data)
    return data

  def close_dist(self):
    if dist.is_initialized():
      dist.barrier()
      dist.destroy_process_group()
    return
  
  @property
  def desc(self):
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
