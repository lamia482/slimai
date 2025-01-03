import sys
import torch
from pathlib import Path
import torch.amp
from torch.distributed.elastic.multiprocessing.errors import record
import mmengine
import slimai
from slimai.helper import help_build, help_utils
from slimai.helper.structure import DataSample
from slimai.helper.utils.dist_env import dist_env


class Runner(object):
  def __init__(self, cfg: mmengine.Config):
    self.cfg = cfg.copy()

    self.work_dir = cfg.work_dir

    self.max_epoch = cfg.RUNNER.max_epoch
    self.gradient_amp = cfg.RUNNER.gradient.get("amp", False)
    self.gradient_accumulation_every_n_steps = cfg.RUNNER.gradient.get("accumulation_every_n_steps", 1)
    self.gradient_scaler = torch.cuda.amp.GradScaler(enabled=self.gradient_amp)

    self.log_level = cfg.RUNNER.logger.get("log_level", "INFO")
    self.log_file = Path(self.work_dir) / cfg.RUNNER.logger.get("log_dir", "logs") / "log.txt"
    self.log_every_n_steps = cfg.RUNNER.logger.get("log_every_n_steps", 10)

    self.ckpt_save_dir = Path(self.work_dir) / cfg.RUNNER.ckpt.get("save_dir", "ckpts")
    self.ckpt_save_every_n_epochs = cfg.RUNNER.ckpt.get("save_every_n_epochs", 1)
    self.ckpt_keep_max = cfg.RUNNER.ckpt.get("keep_max", -1)
    self.ckpt_keep_best = cfg.RUNNER.ckpt.get("keep_best", True)
    self.ckpt_keep_latest = cfg.RUNNER.ckpt.get("keep_latest", True)
    self.ckpt_record_file = self.ckpt_save_dir / "stat.pkl"
    self.ckpt_min_loss = float("inf")
    self.ckpt_best_path = self.ckpt_save_dir / "best.pth"
    self.ckpt_latest_path = self.ckpt_save_dir / "latest.pth"

    help_utils.update_logger(self.log_file, self.log_level)
    help_utils.print_log(f"Work dir: {self.work_dir}")
    slimai.check_env()

    self.train_dataloader, self.valid_dataloader, self.test_dataloader, \
      self.model, self.solver = self.build_components(self.cfg)
    
    self.epoch = 0
    self.load_ckpt(resume=self.cfg.RUNNER.resume.enable, 
                   resume_from=self.cfg.RUNNER.resume.resume_from, 
                   load_from=self.cfg.RUNNER.resume.load_from)
    return
  
  def build_components(self, cfg):
    cfg = cfg.copy()
    train_loader = help_build.build_dataloader(cfg.TRAIN_LOADER)
    valid_loader = help_build.build_dataloader(cfg.VALID_LOADER)
    test_loader = help_build.build_dataloader(cfg.TEST_LOADER)

    cfg.MODEL.metric = cfg.METRIC
    model = help_build.build_model(cfg.MODEL)
    model = dist_env.init_dist(module=model)
    solver = help_build.build_solver(cfg.RUNNER.solver, model)
    dist_env.sync()
    help_utils.print_log("Created Dist runner, desc: {}".format(dist_env.desc), main_process_only=False)
    return train_loader, valid_loader, test_loader, model, solver
  
  @record
  def run(self, *, action):
    assert action in ["train", "infer", "evaluate"]
    action = getattr(self, action)
    return action()
  
  def train(self):
    for epoch in range(self.epoch, self.max_epoch):
      self.train_dataloader.sampler.set_epoch(epoch)
      desc = f"[TRAIN {epoch+1}/{self.max_epoch} EPOCH]" + " {msg}"

      self.model.train()
      self.solver.zero_grad()

      for step, batch_info in enumerate(self.train_dataloader):
        msg = f"Step: {step+1}/{len(self.train_dataloader)}"
        batch_info = DataSample(**batch_info).to(self.model.device)
        batch_data = batch_info.pop("image")

        with torch.autocast(device_type=self.model.device.type, 
                            enabled=self.gradient_amp, dtype=torch.bfloat16):
          loss_dict = self.model(batch_data, batch_info, mode="loss")
          loss_dict = dist_env.sync(loss_dict)

        total_loss, loss_msg, n_loss = 0, "", 0
        for key, loss in loss_dict.items():
          loss_msg += f", {key}: {loss:.4f}"
          if "loss" in key:
            n_loss += 1
            total_loss += loss

        if n_loss > 1:
          msg += f", total_loss: {total_loss:.4f}"
        msg += loss_msg
          
        self.gradient_scaler.scale(total_loss).backward()

        if (step + 1) % self.gradient_accumulation_every_n_steps == 0:
          self.gradient_scaler.step(self.solver)
          self.gradient_scaler.update()
          self.solver.zero_grad()
          self.solver.scheduler.step()

        if (step + 1) % self.log_every_n_steps == 0:
          help_utils.print_log(desc.format(msg=msg), level="INFO")

      ckpt_path = self.save_ckpt(epoch, loss)
      result_file = Path(self.work_dir) / "results" / f"epoch_{epoch}.pkl"
      metrics = self.evaluate(self.valid_dataloader, result_file)

    return
  
  @torch.no_grad()
  def infer(self, dataloader, result_file, ckpt_path=None):
    if ckpt_path is not None:
      self.load_ckpt(resume=False, resume_from=False, load_from=ckpt_path, strict=True)

    self.model.eval()

    pbar = help_utils.ProgressBar(len(dataloader), desc="Infer")

    results = []
    for step, batch_info in enumerate(dataloader):
      batch_info = DataSample(**batch_info).to(self.model.device)
      batch_data = batch_info.pop("image")
      batch_info = self.model(batch_data, batch_info, mode="predict")
      results.extend(batch_info.split_as_list())
      pbar.update()

    results = dist_env.collect(results)

    if dist_env.is_main_process():
      mmengine.dump(results, result_file)

    dist_env.sync()
    return results
  
  @torch.no_grad()
  def evaluate(self, dataloader, result_file):
    if not Path(result_file).exists():
      results = self.infer(dataloader, result_file)
    else:
      results = mmengine.load(result_file)

    metrics = results
    dist_env.sync()
    return metrics

  def save_ckpt(self, epoch, loss):
    """Save latest N checkpoint and link best and latest checkpoint
    """
    ckpt_path = self.ckpt_save_dir / f"epoch_{epoch}.pth"
    if (
      epoch % self.ckpt_save_every_n_epochs == 0
      and dist_env.is_main_process()
    ):
      update_best = False
      if loss < self.ckpt_min_loss:
        self.ckpt_min_loss = loss
        update_best = True and self.ckpt_keep_best

      def _save(ckpt):
        Path(ckpt).resolve().parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model=self.model.state_dict(), 
                        # cfg=self.cfg, 
                        epoch=epoch, loss=loss, min_loss=self.ckpt_min_loss), ckpt)
        return
      
      help_utils.print_log(f"Save checkpoint to {ckpt_path}")
      _save(ckpt_path)

      if self.ckpt_keep_latest:
        self.ckpt_latest_path.unlink(missing_ok=True)
        self.ckpt_latest_path.symlink_to(ckpt_path)

      if update_best:
        _save(self.ckpt_best_path)

      if not self.ckpt_record_file.exists():
        mmengine.dump([], self.ckpt_record_file)

      records = mmengine.load(self.ckpt_record_file)
      records.append(dict(
        ckpt=ckpt_path, epoch=epoch, loss=loss, min_loss=self.ckpt_min_loss
      ))
      if (
        (self.ckpt_keep_max is not None and self.ckpt_keep_max > 0) 
        and (len(records) >= self.ckpt_keep_max)
      ):
        path = Path(records[-self.ckpt_keep_max]["ckpt"]).resolve()
        path.unlink(missing_ok=True)

      mmengine.dump(records, self.ckpt_record_file)

    dist_env.sync()
    return ckpt_path

  def load_ckpt(self, *, resume, resume_from, load_from=None, strict=True):
    """resume or just load from checkpoint, resume_from > load_from
    """
    if resume_from in ["best", "latest"]:
      resume_from = getattr(self, f"ckpt_{resume_from}_path")
    elif isinstance(resume_from, int):
      resume_from = Path(self.ckpt_save_dir) / f"epoch_{resume_from}.pth"
    
    # only resume when ckpt exists
    if resume_from is None or not(Path(resume_from).resolve().exists()):
      resume = False
    else:
      resume = True
      load_from = resume_from

    if load_from is None:
      help_utils.print_log("No checkpoint to load, build from scratch", level="WARNING")
    else:
      load_from = Path(load_from).resolve()
      if not load_from.exists():
        help_utils.print_log(f"Checkpoint {load_from} not found", level="ERROR")
        sys.exit(2)

      help_utils.print_log(f"{'Resume' if resume else 'Load'} checkpoint from {load_from}")
      ckpt = torch.load(load_from, map_location="cpu")
      keys = self.model.load_state_dict(ckpt["model"], strict=(resume or strict))
      if resume:
        #TODO: check cfg
        self.epoch = ckpt["epoch"] + 1 # resume from epoch + 1
        self.ckpt_min_loss = ckpt.get("min_loss", float("inf"))

    return dist_env.sync()
  