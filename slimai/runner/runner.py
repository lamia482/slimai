from torch.distributed.elastic.multiprocessing.errors import record
import mmengine
from mmengine.registry import RUNNERS as MMEngineRUNNERS
from mmengine.runner import Runner as MMEngineRunner


@record
class Runner(object):
  def __init__(self, config: mmengine.Config):
    self.config = config.copy()
    if "runner_type" not in self.config:
      runner = MMEngineRunner.from_cfg(self.config)
    else:
      runner = MMEngineRUNNERS.build(self.config)
    self.runner = runner
    return
  
  def train(self):
    return self.runner.train()
  
  def infer(self):
    return
  
  def evaluate(self):
    return
