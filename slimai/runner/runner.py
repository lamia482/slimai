import time
import torch
import matplotlib
from pathlib import Path
from functools import partial
import mmengine
import slimai
from slimai.helper import (
  help_build, help_utils, DataSample, Distributed, Checkpoint, Gradient
)
from slimai.helper.utils import PytorchNetworkUtils
from torch.distributed.elastic.multiprocessing.errors import record


class Runner(object):
  def __init__(self, cfg: mmengine.Config):

    self.dist = Distributed.create()

    # Initialize runner with configuration
    self.cfg = cfg.copy()

    # Set up working directory
    self.work_dir = Path(cfg.work_dir).resolve()

    # Logger configuration
    logger = cfg.RUNNER.logger    
    help_utils.update_logger(
      self.work_dir / logger.get("log_dir", "logs") / f"{int(time.time())}.txt", 
      logger.get("log_level", "INFO"))
    
    self.log_every_n_steps = logger.get("log_every_n_steps", 10)

    # Check environment
    slimai.check_env()

    # Runner configuration
    self.max_epoch = cfg.RUNNER.max_epoch

    # Initialize gradient handler
    gradient = cfg.RUNNER.gradient
    self.gradient = Gradient(
      amp=gradient.get("amp", False),
      accumulation_every_n_steps=gradient.get("accumulation_every_n_steps", 1),
      clip=gradient.get("clip", None)
    )

    # Initialize checkpoint handler
    ckpt = cfg.RUNNER.ckpt
    self.checkpoint = Checkpoint(
      save_dir=self.work_dir / ckpt.get("save_dir", "ckpts"),
      save_every_n_epochs=ckpt.get("save_every_n_epochs", 1),
      keep_max=ckpt.get("keep_max", -1),
      keep_best=ckpt.get("keep_best", True),
      keep_latest=ckpt.get("keep_latest", True)
    )

    self.eval_every_n_epochs = ckpt.get("eval_every_n_epochs", 1)

    # Dump config to work_dir
    self.dump_cfg()

    # Build components like dataloaders, model, solver, and metric
    train_dataloader, valid_dataloader, test_dataloader, \
      arch, model, solver, scheduler, loss, metric = self.build_components(self.cfg)
    
    # Initialize epoch and load checkpoint if needed
    resume = cfg.RUNNER.resume
    model, solver, scheduler, ckpt = self.checkpoint.load(
      model=model,
      solver=solver,
      scheduler=scheduler,
      resume=resume.enable,
      resume_from=resume.resume_from,
      load_from=resume.load_from
    )
    self.epoch = ckpt.get("epoch", 0)

    # prepare model and solver for distributed training
    self.train_dataloader, self.valid_dataloader, self.test_dataloader, self.arch, \
      self.model, self.solver, self.scheduler, self.loss, self.metric = self.dist.prepare_for_distributed(
        train_dataloader, valid_dataloader, test_dataloader, arch, \
          model, solver, scheduler, loss, metric)
    return
  
  def build_components(self, cfg):
    """Build dataloader, arch, model, solver, metric"""
    cfg = cfg.copy()
    
    train_loader, valid_loader, test_loader = list(map(help_build.build_dataloader, [
      cfg.TRAIN_LOADER, 
      cfg.get("VALID_LOADER", cfg.get("VAL_LOADER", dict())), 
      cfg.get("TEST_LOADER", dict())
    ]))

    arch = help_build.build_model(cfg.MODEL)
    arch.compile(cfg.RUNNER.get("compile", False))
    arch.checkpointing(cfg.RUNNER.get("checkpointing", True))
    
    model = arch.model
    solver = arch.solver
    scheduler = arch.scheduler
    loss = arch.loss

    # Log model parameter size
    help_utils.print_log(f"Model({arch.__class__.__name__}) built successfully, "
                         f"{PytorchNetworkUtils.desc(model)}")

    metric = help_build.build_metric(cfg.get("METRIC", dict()))

    self.dist.env.sync()
    help_utils.print_log("Created runner, desc: {}, {}".format(
      self.dist, self.dist.env.desc), main_process_only=False)
      
    return train_loader, valid_loader, test_loader, \
           arch, model, solver, scheduler, loss, metric
  
  @record
  def run(self, *, action):
    """Run the runner, action can be "train", "infer", "evaluate"."""
    assert action in ["train", "infer", "evaluate"]
    if action == "train":
      return self.train()
    elif action == "infer":
      return self.infer(self.test_dataloader, self.work_dir / "results" / "test.pkl")
    elif action == "evaluate":
      return self.evaluate(self.test_dataloader, self.work_dir / "results" / "test.pkl")
    raise ValueError(f"Invalid action: {action}")
  
  def step_train(self, i_step, total_steps, 
                 batch_data, batch_info):
    """Train the model for one step."""
    accumulation_i_step = i_step % self.gradient.accumulation_every_n_steps

    train_forward_func = partial(self.arch, mode="loss")

    with torch.autocast(device_type=self.arch.device.type, 
                        enabled=self.gradient.amp, dtype=self.dist.mix_dtype):
      loss_dict = train_forward_func(batch_data, batch_info)

      total_loss = None
      for loss_name, loss_value in loss_dict.items():
        if "loss" in loss_name:
          if total_loss is None:
            total_loss = torch.tensor(0.0, device=self.arch.device)
          total_loss += loss_value
      assert (
        total_loss is not None
      ), "total_loss must be provided"
      
      self.gradient.step(
        model=self.model,
        solver=self.solver,
        scheduler=self.scheduler,
        loss=total_loss,
        i_step=i_step,
        total_steps=total_steps,
        accumulation_i_step=accumulation_i_step
      )

    return total_loss, loss_dict
  
  def train(self):
    """Train the model."""
    assert (
      self.train_dataloader is not None
    ), "train_dataloader must be provided"

    for self.epoch in range(self.epoch, self.max_epoch):
      self.train_dataloader.sampler.set_epoch(self.epoch)

      self.epoch += 1 # Increment epoch for checkpoint saving
      desc = f"[TRAIN {self.epoch}/{self.max_epoch} EPOCH]" + " {msg}"

      # before epoch
      self.arch.epoch_precede_hooks(runner=self)
      
      # walk through one epoch
      avg_loss = 0.0
      for self.step, batch_info in enumerate(self.train_dataloader):
        msg = f"Step: {self.step+1}/{len(self.train_dataloader)}"
        batch_info = self.dist.prepare_for_distributed(batch_info)
        batch_info = DataSample(**batch_info).to(self.arch.device)
        batch_data = batch_info.pop("image")

        # before forward step
        self.arch.step_precede_hooks(runner=self)

        # forward step
        total_loss, loss_dict = self.step_train(self.step, len(self.train_dataloader), 
                                                batch_data, batch_info)
        
        # update avg loss
        avg_loss = (avg_loss * self.step + total_loss.detach().cpu().item()) / (self.step + 1)

        msg += ", " + ", ".join([
          f"lr: {self.scheduler.get_last_lr()[0]:.6f}", 
          f"avg loss: {avg_loss:.6f}", 
          *[f"{key}: {loss:.6f}" for key, loss in loss_dict.items()],
        ])

        if (self.step + 1) % self.log_every_n_steps == 0:
          help_utils.print_log(desc.format(msg=msg), level="INFO")

        # after forward step
        self.arch.step_succeed_hooks(runner=self)

      # after epoch
      self.arch.epoch_succeed_hooks(runner=self)
      
      # Evaluate on validation dataset
      if (self.epoch % self.eval_every_n_epochs == 0) and (self.valid_dataloader is not None):
        result_file = self.work_dir / "results" / f"epoch_{self.epoch}.pkl"
        eval_metrics = self.evaluate(self.valid_dataloader, result_file)
        avg_loss = eval_metrics.get("loss", avg_loss)

      # Save checkpoint with strategy
      self.checkpoint.save(
        self.model, 
        self.solver, 
        self.scheduler,
        self.epoch, 
        avg_loss, 
        cfg=self.cfg.MODEL)

    return
  
  @torch.no_grad()
  def infer(self, dataloader, result_file):
    """Infer on dataloader."""
    assert (
      dataloader is not None
    ), "dataloader must be provided for infer"

    self.model.eval()

    pbar = help_utils.ProgressBar(len(dataloader), desc="Infer")
    infer_forward_func = partial(self.arch, mode="predict")

    results = []
    for step, batch_info in enumerate(dataloader):
      batch_info = self.dist.prepare_for_distributed(batch_info)
      batch_info = DataSample(**batch_info).to(self.arch.device)
      batch_data = batch_info.pop("image")
      with torch.autocast(device_type=self.arch.device.type, 
                          enabled=self.gradient.amp, dtype=self.dist.mix_dtype):
        batch_info = infer_forward_func(batch_data, batch_info).cpu()
      results.extend(batch_info.split_as_list())
      pbar.update(sep="\r\t")
    pbar.close()

    help_utils.print_log(f"Collecting data from all nodes...")
    results = self.dist.env.collect(results)
    results = dict(batch_info=results)

    if self.dist.env.is_main_process():
      help_utils.print_log(f"Dump infer result into: {result_file}")
      mmengine.dump(results, result_file)

    self.dist.env.sync()
    return results
  
  @torch.no_grad()
  def evaluate(self, dataloader, result_file):
    """Evaluate on result_file, if not exists, infer first with dataloader."""
    if not Path(result_file).exists():
      results = self.infer(dataloader, result_file)
    else:
      results = mmengine.load(result_file)

    metrics = None

    if self.dist.env.is_main_process():
      batch_info = results["batch_info"]
      # for better performance, move to arch device first
      merge_result = DataSample.merge_from_list(batch_info).to(self.arch.device)
      output = merge_result.output
      targets = {key: getattr(merge_result, key) for key in dataloader.dataset.ann_keys}
      metrics = self.metric(output, targets)

      # split figure and metric
      msg_list = []
      for key, fig in metrics.items():
        if isinstance(fig, matplotlib.figure.Figure):
          fig.savefig(str(result_file).replace(".pkl", f"_{key}.png"))
        else:
          msg_list.append(f"{key}: {fig:.6f}")

      help_utils.print_log(f"Metrics: {', '.join(msg_list)}")
      results["metrics"] = DataSample(**metrics).to("cpu").to_dict() # move to cpu to dump
      help_utils.print_log(f"Dump metric into: {result_file}")
      mmengine.dump(results, result_file)

    metrics = self.dist.env.broadcast(metrics)
    return metrics

  def dump_cfg(self):
    """Dump config and work dir."""
    try:
      mmengine.Config.dump(self.cfg, self.work_dir / "config.py")
    except Exception as e:
      help_utils.print_log(f"Error dumping config: {e}", level="ERROR")
      help_utils.print_log(f"Please check the config file", level="ERROR")
      exit(2)

    help_utils.print_log(f"Config: \n{self.cfg.dump()}")
    help_utils.print_log(f"Work dir: {self.work_dir}")
    return
