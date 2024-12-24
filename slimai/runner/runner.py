from torch.distributed.elastic.multiprocessing.errors import record
import mmengine
from slimai.helper import help_build, help_utils
from slimai.helper.structure import DataSample
from slimai.helper.utils.dist_env import dist_env


class Runner(object):
  def __init__(self, cfg: mmengine.Config):
    self.cfg = cfg.copy()
    self.train_dataloader, self.valid_dataloader, self.test_dataloader, \
      self.model, self.solver = self.build_components(self.cfg)
    return
  
  def build_components(self, cfg):
    train_loader = help_build.build_dataloader(cfg.TRAIN_LOADER)
    valid_loader = help_build.build_dataloader(cfg.VALID_LOADER)
    test_loader = help_build.build_dataloader(cfg.TEST_LOADER)
    model = help_build.build_model(cfg.MODEL)
    model = dist_env.init_dist(model)
    solver = help_build.build_solver(cfg.SOLVER, model)
    help_utils.print_log("Created Dist runner, desc: {}".format(dist_env.desc), main_process_only=False)
    return train_loader, valid_loader, test_loader, model, solver
  
  @record
  def run(self, *, action):
    assert action in ["train", "infer", "evaluate"]
    action = getattr(self, action)
    return action()
  
  def train(self):
    self.model.train()

    max_epoch = 100
    for epoch in range(max_epoch):
      self.train_dataloader.sampler.set_epoch(epoch)
      desc = f"[TRAIN {epoch+1}/{max_epoch} EPOCH]" + " {msg}"
      for step, batch_info in enumerate(self.train_dataloader):
        batch_info = DataSample(**batch_info).to(self.model.device)
        batch_data = batch_info.pop("image")
        self.solver.zero_grad()
        loss, kappa = self.model(batch_data, batch_info, mode="loss")
        help_utils.print_log(f"{dist_env.desc}, Loss: {loss}, Kappa: {kappa}", main_process_only=False, level="NOTSET")
        loss = dist_env.sync(loss)
        kappa = dist_env.sync(kappa)
        help_utils.print_log(f"Synced AVG Loss: {loss}, AVG Kappa: {kappa}", main_process_only=False, level="NOTSET")
        loss.backward()
        self.solver.step()

        help_utils.print_log(desc.format(
          msg=f"Step: {step+1}/{len(self.train_dataloader)}, Loss: {loss:.6f}, Kappa: {kappa:.6f}"
          ), level="INFO")
    return
  
  def infer(self):
    return
  
  def evaluate(self):
    return
  
  def __del__(self):
    dist_env.close_dist()
    return
  
