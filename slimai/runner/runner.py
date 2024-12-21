from torch.distributed.elastic.multiprocessing.errors import record
import mmengine
from slimai.helper import help_build


class Runner(object):
  def __init__(self, cfg: mmengine.Config):
    self.cfg = cfg.copy()
    self.build_pipeline(self.cfg)
    return
  
  def build_pipeline(self, cfg):
    train_loader = help_build.build_dataloader(cfg.TRAIN_LOADER)
    valid_loader = help_build.build_dataloader(cfg.VALID_LOADER)
    test_loader = help_build.build_dataloader(cfg.TEST_LOADER)
    model = help_build.build_model(cfg.MODEL)
    return

  @record
  def run(self, *, action):
    assert action in ["train", "infer", "evaluate"]
    action = getattr(self, action)
    return action()
  
  def train(self):
    return
  
  def infer(self):
    return
  
  def evaluate(self):
    return
  