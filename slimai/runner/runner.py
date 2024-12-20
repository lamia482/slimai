from torch.distributed.elastic.multiprocessing.errors import record
import mmengine


class Runner(object):
  def __init__(self, cfg: mmengine.Config):
    self.cfg = cfg.copy()
    self.build_pipeline(self.cfg)
    return
  
  def build_pipeline(self, cfg):
    train_loader = self.build_dataloader(cfg.train_loader)
    valid_loader = self.build_dataloader(cfg.valid_loader)
    test_loader = self.build_dataloader(cfg.test_loader)
    model = self.build_model(cfg.model)
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
  
  def build_dataloader(self, cfg):
    return
  
  def build_model(self, cfg):
    return