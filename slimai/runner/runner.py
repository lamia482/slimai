import sys
import time
import torch
from pathlib import Path
import torch.amp
from torch.distributed.elastic.multiprocessing.errors import record
import mmengine
import slimai
from slimai.helper import help_build, help_utils
from slimai.helper.structure import DataSample
from slimai.helper.utils.dist_env import dist_env
from slimai.helper.utils.network import PytorchNetworkUtils


class Runner(object):
  def __init__(self, cfg: mmengine.Config):
    # Initialize runner with configuration
    self.cfg = cfg.copy()

    # Set up working directory
    self.work_dir = Path(cfg.work_dir).resolve()

    # Runner configuration
    self.max_epoch = cfg.RUNNER.max_epoch
    self.gradient_amp = cfg.RUNNER.gradient.get("amp", False) and torch.cuda.is_bf16_supported()
    self.gradient_accumulation_every_n_steps = cfg.RUNNER.gradient.get("accumulation_every_n_steps", 1)
    assert (
      self.gradient_accumulation_every_n_steps >= 1
    ), "gradient_accumulation_every_n_steps must be greater than or equal to 1, but got {}".format(self.gradient_accumulation_every_n_steps)
    assert (
      self.gradient_accumulation_every_n_steps == 1
    ), "gradient accumulation is not supported yet"
    self.gradient_scaler = torch.cuda.amp.GradScaler(enabled=self.gradient_amp)
    self.gradient_clip = cfg.RUNNER.gradient.get("clip", None)

    # Logger configuration
    self.log_level = cfg.RUNNER.logger.get("log_level", "INFO")
    self.log_file = self.work_dir / cfg.RUNNER.logger.get("log_dir", "logs") / f"{int(time.time())}.txt"
    self.log_every_n_steps = cfg.RUNNER.logger.get("log_every_n_steps", 10)

    # Checkpoint configuration
    self.ckpt_save_dir = self.work_dir / cfg.RUNNER.ckpt.get("save_dir", "ckpts")
    self.ckpt_save_every_n_epochs = cfg.RUNNER.ckpt.get("save_every_n_epochs", 1)
    self.ckpt_keep_max = cfg.RUNNER.ckpt.get("keep_max", -1)
    self.ckpt_keep_best = cfg.RUNNER.ckpt.get("keep_best", True)
    self.ckpt_keep_latest = cfg.RUNNER.ckpt.get("keep_latest", True)
    self.ckpt_record_file = self.ckpt_save_dir / "stat.pkl"
    self.ckpt_min_loss = float("inf") # will be updated in `load_ckpt`
    self.ckpt_best_path = self.ckpt_save_dir / "best.pth"
    self.ckpt_latest_path = self.ckpt_save_dir / "latest.pth"
    self.eval_every_n_epochs = cfg.RUNNER.ckpt.get("eval_every_n_epochs", 1)
    # Initialize logger and environment
    help_utils.update_logger(self.log_file, self.log_level)    
    slimai.check_env()
    self.dump_cfg()

    # Build components like dataloaders, model, solver, and metric
    self.train_dataloader, self.valid_dataloader, self.test_dataloader, \
      self.arch, self.model, self.solver, self.metric = self.build_components(self.cfg)
    
    # Initialize epoch and load checkpoint if needed
    self.epoch = 0 # will be updated in `load_ckpt`
    self.load_ckpt(self.model, self.solver, 
                   resume=self.cfg.RUNNER.resume.enable, 
                   resume_from=self.cfg.RUNNER.resume.resume_from, 
                   load_from=self.cfg.RUNNER.resume.load_from)
    return
  
  def build_components(self, cfg):
    """Build dataloader, arch, model, solver, metric"""
    cfg = cfg.copy()
    
    train_loader = help_build.build_dataloader(cfg.TRAIN_LOADER)
    valid_loader = help_build.build_dataloader(cfg.get("VALID_LOADER", dict()))
    test_loader = help_build.build_dataloader(cfg.get("TEST_LOADER", dict()))

    arch = help_build.build_model(cfg.MODEL)
    model = arch.model
    solver = arch.solver

    metric = help_build.build_metric(cfg.METRIC)

    dist_env.sync()
    help_utils.print_log("Created runner, desc: {}".format(dist_env.desc), main_process_only=False)
    return train_loader, valid_loader, test_loader, arch, model, solver, metric
  
  @record
  def run(self, *, action):
    """Run the runner, action can be "train", "infer", "evaluate"."""
    assert action in ["train", "infer", "evaluate"]
    action = getattr(self, action)
    return action()
  
  def step_train(self, i_step, total_steps, 
                 grad_accumulation_every_n_steps, 
                 batch_data, batch_info):
    """Train the model for one step."""
    if i_step == 0: # clear grad before first step
      self.solver.zero_grad()

    accumulation_i_step = i_step % grad_accumulation_every_n_steps

    with torch.autocast(device_type=self.arch.device.type, 
                        enabled=self.gradient_amp, dtype=torch.bfloat16):
      if (grad_accumulation_every_n_steps == 1 # no grad accumulation mode
          ) or (i_step == total_steps - 1 # last step to accumulate grad
          ) or (accumulation_i_step == grad_accumulation_every_n_steps - 1 # meet accumulation steps
          ):
        loss_dict = self.arch(batch_data, batch_info, mode="loss")
        total_loss = sum(loss_dict.values())

        # Scale loss with AMP mode and backward
        self.gradient_scaler.scale(total_loss / grad_accumulation_every_n_steps).backward()
        
        if self.gradient_clip is not None and self.gradient_scaler.is_enabled():
          self.gradient_scaler.unscale_(self.solver)
        PytorchNetworkUtils.clip_gradients(self.model, self.gradient_clip)

        # Step optimizer and update learning rate after gradient accumulation
        self.gradient_scaler.step(self.solver)
        self.gradient_scaler.update()
        self.solver.zero_grad()
        self.solver.scheduler.step() # Scheduler is created when building solver
      else:
        # BUG: crash here
        with self.arch.no_sync():
          loss_dict = self.arch(batch_data, batch_info, mode="loss")
        total_loss = sum(loss_dict.values())
        # Scale loss with AMP mode and backward
        self.gradient_scaler.scale(total_loss / grad_accumulation_every_n_steps).backward()

    return total_loss, loss_dict
  
  def train(self):
    """Train the model."""
    for self.epoch in range(self.epoch, self.max_epoch):
      self.train_dataloader.sampler.set_epoch(self.epoch)

      self.epoch += 1 # Increment epoch for checkpoint saving
      desc = f"[TRAIN {self.epoch}/{self.max_epoch} EPOCH]" + " {msg}"

      # before epoch
      self.arch.epoch_precede_hooks(runner=self)
      
      for self.step, batch_info in enumerate(self.train_dataloader):
        msg = f"Step: {self.step+1}/{len(self.train_dataloader)}, Global Rank: {dist_env.global_rank}"
        batch_info = DataSample(**batch_info).to(self.arch.device)
        batch_data = batch_info.pop("image")

        # before forward step
        self.arch.step_precede_hooks(runner=self)

        # forward step
        total_loss, loss_dict = self.step_train(self.step, len(self.train_dataloader), 
                                                self.gradient_accumulation_every_n_steps, 
                                                batch_data, batch_info)
        if len(loss_dict) > 1:
          msg += f", total_loss: {total_loss:.6f}"
        msg += ", " + ", ".join([f"{key}: {loss:.6f}" for key, loss in loss_dict.items()])
        msg += f", lr: {self.solver.scheduler.get_last_lr()[0]:.6f}"

        if (self.step + 1) % self.log_every_n_steps == 0:
          help_utils.print_log(desc.format(msg=msg), level="INFO")

        # after forward step
        self.arch.step_succeed_hooks(runner=self)

      # after epoch
      self.arch.epoch_succeed_hooks(runner=self)
      
      # Save checkpoint with strategy
      self.save_ckpt(self.model, self.solver, self.epoch, total_loss)

      # Evaluate on validation dataset
      if (self.epoch % self.eval_every_n_epochs == 0) and (self.valid_dataloader is not None):
        result_file = self.work_dir / "results" / f"epoch_{self.epoch}.pkl"
        self.evaluate(self.valid_dataloader, result_file)

    return
  
  @torch.no_grad()
  def infer(self, dataloader, result_file, ckpt_path=None):
    """Infer on dataloader, if ckpt_path is not None, load ckpt from ckpt_path."""
    if ckpt_path is not None:
      self.load_ckpt(self.model, self.solver, 
                     resume=False, resume_from=False, load_from=ckpt_path, strict=True)

    self.model.eval()

    pbar = help_utils.ProgressBar(len(dataloader), desc="Infer")

    results = []
    for step, batch_info in enumerate(dataloader):
      batch_info = DataSample(**batch_info).to(self.arch.device)
      batch_data = batch_info.pop("image")
      batch_info = self.arch(batch_data, batch_info, mode="predict").cpu()
      results.extend(batch_info.split_as_list())
      pbar.update(sep="\r\t")
    pbar.close()

    help_utils.print_log(f"Collecting data from all nodes...")
    results = dist_env.collect(results)
    results = dict(batch_info=results)

    if dist_env.is_main_process():
      help_utils.print_log(f"Dump infer result into: {result_file}")
      mmengine.dump(results, result_file)

    dist_env.sync()
    return results
  
  @torch.no_grad()
  def evaluate(self, dataloader, result_file):
    """Evaluate on result_file, if not exists, infer first with dataloader."""
    if not Path(result_file).exists():
      results = self.infer(dataloader, result_file)
    else:
      results = mmengine.load(result_file)

    batch_info = results["batch_info"]
    logits = torch.stack([result.output for result in batch_info])
    targets = torch.stack([result.label for result in batch_info])
    metrics = self.metric(logits, targets)

    if dist_env.is_main_process():
      help_utils.print_log(f"Metrics: {', '.join([f'{key}: {value:.6f}' for key, value in metrics.items()])}")
      results["metrics"] = metrics
      help_utils.print_log(f"Dump metric result into: {result_file}")
      mmengine.dump(results, result_file)

    dist_env.sync()
    return metrics

  def save_ckpt(self, model, solver, epoch, loss):
    """Save latest N checkpoint and link best and latest checkpoint."""
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
        no_ddp_weight = PytorchNetworkUtils.fix_weight(model.state_dict(), to_ddp=False)
        torch.save(dict(model=self.cfg.MODEL, 
                        weight=no_ddp_weight, # default save non-ddp weight
                        solver=solver.state_dict(),
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

  def load_ckpt(self, model=None, solver=None, *, resume=True, resume_from=None, load_from=None, strict=True):
    """Resume or just load from checkpoint, resume_from > load_from."""
    if resume_from in ["best", "latest"]:
      resume_from = getattr(self, f"ckpt_{resume_from}_path")
    elif isinstance(resume_from, int):
      resume_from = Path(self.ckpt_save_dir) / f"epoch_{resume_from}.pth"
    
    # Only resume when checkpoint exists
    if resume_from is None or not(Path(resume_from).resolve().exists()):
      resume = False
    else:
      resume = True
      load_from = resume_from

    if load_from is None:
      help_utils.print_log("No checkpoint to load, build from scratch", level="WARNING")
      assert (
        model is not None
      ), "model must be provided when no checkpoint to load"
    else:
      load_from = Path(load_from).resolve()
      if not load_from.exists():
        help_utils.print_log(f"Checkpoint {load_from} not found", level="ERROR")
        sys.exit(2)

      help_utils.print_log(f"{'Resume' if resume else 'Load'} checkpoint from {load_from}")
      ckpt = torch.load(load_from, map_location="cpu")

      if model is None:
        model = help_build.build_model(ckpt["model"])        
        if dist_env.is_dist_initialized(): # adapt model weight to ddp if necessary
          ckpt["weight"] = PytorchNetworkUtils.fix_weight(ckpt["weight"], to_ddp=True)

      model.load_state_dict(ckpt["weight"], strict=(resume or strict))
      
      if solver is not None:
        solver.load_state_dict(ckpt["solver"])

      if resume:
        #TODO: check cfg
        self.epoch = ckpt["epoch"]
        self.ckpt_min_loss = ckpt.get("min_loss", float("inf"))

    return model
  
  def dump_cfg(self):
    """Dump config and work dir."""
    help_utils.print_log(f"Work dir: {self.work_dir}")
    help_utils.print_log(f"Config: {self.cfg}")
    # mmengine.Config.dump(self.cfg, self.work_dir / "config.py")
    return