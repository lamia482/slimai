import time
import torch
import shutil
import matplotlib
from pathlib import Path
from functools import partial
from typing import Dict, Any, Optional
import mmengine
import slimai
from slimai.helper import (
  help_build, help_utils, DataSample, Distributed, Checkpoint, Gradient, Record
)
from slimai.helper.utils import PytorchNetworkUtils, recursive_apply
from torch.distributed.elastic.multiprocessing.errors import record
from .report import ExperimentReporter


class Runner(object):
  def __init__(self, cfg: mmengine.Config):

    self.record = Record(cfg=cfg)
    self.dist = Distributed()

    # Initialize runner with configuration
    self.cfg = cfg.copy()

    # Set up working directory
    self.work_dir = Path(cfg.work_dir).resolve()
    
    # Logger configuration
    logger = cfg.RUNNER.logger
    self.log_dir = self.work_dir / logger.get("log_dir", "logs")
    help_utils.update_logger(self.log_dir / f"{int(time.time())}.txt", 
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
      save_every_n_steps=ckpt.get("save_every_n_steps", None),
      save_every_n_epochs=ckpt.get("save_every_n_epochs", 1),
      keep_max=ckpt.get("keep_max", -1),
      keep_best=ckpt.get("keep_best", True),
      keep_latest=ckpt.get("keep_latest", True), 
      save_on_rank_0=cfg.RUNNER.get("save_on_rank_0", True)
    )

    self.eval_every_n_epochs = ckpt.get("eval_every_n_epochs", 1)
    self.eval_every_n_steps = ckpt.get("eval_every_n_steps", None)
    self.eval_every_n_epochs = self.eval_every_n_epochs or 1
    self.eval_every_n_steps = self.eval_every_n_steps or 0
    self.epoch_records = []
    self.best_valid_loss = float("inf")
    self.best_valid_epoch = None
    self.best_valid_ckpt_path = self.checkpoint.save_dir / "best_valid.pth"
    self.best_train_ckpt_path = self.checkpoint.best_path
    self.last_epoch_ckpt_path = None
    self.last_step_ckpt_path = None

    # Dump config to work_dir
    self.archive_env_for_reproducibility()

    # Build components like dataloaders, model, solver, and metric
    train_dataloader, valid_dataloader, test_dataloader, external_test_dataloaders, \
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
    self.step = ckpt.get("step", 0)
    self.epoch = ckpt.get("epoch", 0)
    self.train_avg_loss = ckpt.get("loss", None) or 0.0

    # prepare model and solver for distributed training
    self.train_dataloader, self.valid_dataloader, self.test_dataloader, self.external_test_dataloaders, \
    self.arch, self.model, self.solver, self.scheduler, self.loss, self.metric = \
      self.dist.prepare_for_distributed(train_dataloader, valid_dataloader, test_dataloader, external_test_dataloaders, \
                                        arch, model, solver, scheduler, loss, metric)
    self.reporter = ExperimentReporter(
      work_dir=self.work_dir,
      cfg=self.cfg,
      train_dataloader=self.train_dataloader,
      valid_dataloader=self.valid_dataloader,
      test_dataloader=self.test_dataloader,
      external_test_dataloaders=self.external_test_dataloaders,
      model_desc=self.model_desc,
    )
    return
  
  def build_components(self, cfg):
    """Build dataloader, arch, model, solver, metric"""
    cfg = cfg.copy()
    
    train_loader, valid_loader, test_loader = list(map(help_build.build_dataloader, [
      cfg.TRAIN_LOADER, 
      cfg.get("VALID_LOADER", cfg.get("VAL_LOADER", dict())), 
      cfg.get("TEST_LOADER", dict())
    ]))
    external_loader_cfgs = cfg.get("EXTERNAL_TEST_LOADERS", None)
    external_test_loaders = {}
    if isinstance(external_loader_cfgs, dict):
      for external_name, external_loader_cfg in external_loader_cfgs.items():
        if external_loader_cfg is None:
          continue
        external_loader = help_build.build_dataloader(external_loader_cfg)
        if external_loader is not None:
          external_test_loaders[external_name] = external_loader

    arch = help_build.build_model(cfg.MODEL)
    arch.compile(cfg.RUNNER.get("compile", False))
    arch.checkpointing(cfg.RUNNER.get("checkpointing", True)) # type: ignore
    
    model, solver, scheduler, loss = arch.extract() # type: ignore

    # Log model parameter size
    self.model_desc = PytorchNetworkUtils.desc(model)
    help_utils.print_log(f"Model({arch.__class__.__name__}) built successfully, "
                         f"{self.model_desc}")

    metric = help_build.build_metric(cfg.get("METRIC", dict()))

    self.dist.env.sync()
    help_utils.print_log("Created runner, desc: {}, {}".format(
      self.dist, self.dist.env.desc), main_process_only=False)
      
    return train_loader, valid_loader, test_loader, external_test_loaders, \
           arch, model, solver, scheduler, loss, metric
  
  @record
  def run(self, *, action):
    """Run the runner, action can be "train", "infer", "evaluate"."""
    assert (
      action in ["train", "infer", "evaluate"]
    ), f"Invalid action: {action}"

    if action == "train":
      return self.train()
    elif action == "infer":
      return self.infer(self.test_dataloader, self.work_dir / "results" / "test.pkl")
    elif action == "evaluate":
      return self.evaluate(self.test_dataloader, self.work_dir / "results" / "test.pkl")
    raise ValueError(f"Invalid action: {action}")
  
  def step_train(self, i_step, total_steps, 
                 batch_data, batch_info, phase):
    """Train the model for one step."""

    step_train_start_time = time.time()
    accumulation_i_step = i_step % self.gradient.accumulation_every_n_steps

    train_forward_func = partial(self.arch, mode="loss")

    with torch.autocast(device_type=self.dist.env.accelerator, 
                        enabled=self.gradient.amp, dtype=self.dist.mix_dtype):

      forward_start_time = time.time()
      output, loss_dict = train_forward_func(batch_data, batch_info)
      forward_latency = time.time() - forward_start_time

      total_loss = None
      loss_keys = list(loss_dict.keys())
      for loss_name in loss_keys:
        loss_value = loss_dict[loss_name]
        if "loss" in loss_name:
          if total_loss is None:
            total_loss = torch.zeros_like(loss_value)
          loss_dict[f"{loss_name}^"] = loss_dict.pop(loss_name) # add ^ to loss name for visualization
          total_loss += loss_value
          
      assert (
        total_loss is not None
      ), "losses name should contain 'loss' and so to go backward, please check your loss name, backward loss shows with hat '^'."
      
      backward_start_time = time.time()
      self.gradient.step(
        model=self.model,
        solver=self.solver,
        scheduler=self.scheduler,
        loss=total_loss,
        i_step=i_step,
        total_steps=total_steps,
        accumulation_i_step=accumulation_i_step
      )
      backward_latency = time.time() - backward_start_time

    # log batch sample in training duration
    batch_visual_start_time = time.time()
    if self.record.check_visualize_batch(output, self.step):
      with torch.inference_mode():
        output = getattr(self.arch, "postprocess")(output, batch_info).output
        targets = {key: getattr(batch_info, key) for key in self.train_dataloader.dataset.ann_keys}
      self.record.log_batch_sample(batch_data, output, targets, 
                                    class_names=self.train_dataloader.dataset.class_names,
                                    phase=phase, progress_bar=False, step=self.global_step)
    batch_visual_latency = time.time() - batch_visual_start_time

    step_train_latency = time.time() - step_train_start_time
    latency_dict = dict(
      step_train_latency=step_train_latency,
      forward_latency=forward_latency,
      backward_latency=backward_latency,
      batch_visual_latency=batch_visual_latency,
    )
    return total_loss, loss_dict, latency_dict

  def extract_batch_info(self, batch_info):
    return batch_info.pop("image"), batch_info.get("meta"), batch_info.pop("latency")

  def train(self):
    """Train the model."""
    assert (
      self.train_dataloader is not None
    ), "train_dataloader must be provided"

    for self.epoch in range(self.epoch, self.max_epoch):
      phase = "train"
      desc = f"[TRAIN {self.epoch}/{self.max_epoch} EPOCH]" + " {msg}"

      # before epoch # set training status and epoch
      self.arch.epoch_precede_hooks(runner=self)
      
      # walk through one epoch
      num_steps_per_epoch = len(self.train_dataloader)
      dataloader_generator = iter(self.train_dataloader)
      for self.step in range(self.step, num_steps_per_epoch):
        batch_info = next(dataloader_generator)
        self.global_step = self.step + (self.epoch) * num_steps_per_epoch

        msg = f"Step: {self.step+1}/{num_steps_per_epoch}"
        batch_info = self.dist.prepare_for_distributed(batch_info)
        batch_info = DataSample(**batch_info).to(self.dist.env.device)
        batch_data, batch_meta, batch_latency = self.extract_batch_info(batch_info)

        # before forward step # set steps and epochs, and extra info
        self.arch.step_precede_hooks(runner=self, meta=batch_meta, latency=batch_latency)

        # forward step
        total_loss, loss_dict, latency_dict = self.step_train(
          self.step, num_steps_per_epoch, batch_data, batch_info, phase)
        
        # update avg loss
        total_loss_value = total_loss.detach()
        self.train_avg_loss = (self.train_avg_loss * self.step + total_loss_value) / (self.step + 1)

        log_data = {
          "lr": self.scheduler.get_last_lr()[0],
          "avg_loss": self.train_avg_loss,
          "total_loss": total_loss_value, 
          **loss_dict, 
          **latency_dict, 
          **recursive_apply(lambda x: x.mean(), batch_latency), # type: ignore
          **batch_meta, 
        }
        _, log_msg = self.record.format(log_data)
        msg += log_msg

        self.record.log_step_data(log_data, phase=phase, step=self.global_step)
        if (self.step + 1) % self.log_every_n_steps == 0:
          help_utils.print_log(desc.format(msg=msg), level="INFO")

        # after forward step # save checkpoint and evaluate by step strategy
        self.arch.step_succeed_hooks(runner=self)

      # after epoch # save checkpoint and evaluate by epoch strategy, reset avg loss and step
      self.arch.epoch_succeed_hooks(runner=self)
    self.evaluate_test_with_best_valid()
    return
  
  @torch.inference_mode()
  def infer(self, dataloader, result_file):
    """Infer on dataloader."""
    assert (
      dataloader is not None
    ), "dataloader must be provided for infer"

    self.model.eval()

    pbar = help_utils.ProgressBar(len(dataloader), desc="Infer")
    infer_forward_func = partial(self.arch, mode="predict")

    results = []
    for batch_info in dataloader:
      batch_info = self.dist.prepare_for_distributed(batch_info)
      batch_info = DataSample(**batch_info).to(self.dist.env.device)
      batch_data, batch_meta, batch_latency = self.extract_batch_info(batch_info)

      self.arch.set_extra_attributes(meta=batch_meta, latency=batch_latency)

      with torch.autocast(device_type=self.dist.env.accelerator, 
                          enabled=self.gradient.amp, dtype=self.dist.mix_dtype):
        batch_info = infer_forward_func(batch_data, batch_info).cpu()

      results.extend(batch_info.split_as_list())
      pbar.update(sep="\r\t")
    pbar.close()

    help_utils.print_log(f"Collecting data from all nodes...")
    results = self.dist.env.collect(results)
    results = dict(batch_info=results)

    if self.dist.env.is_main_process():
      Path(result_file).parent.mkdir(parents=True, exist_ok=True)
      help_utils.print_log(f"Dump infer result into: {result_file}")
      mmengine.dump(results, result_file)

    self.dist.env.sync()
    return results

  def _tensor_to_python(self, value):
    if isinstance(value, torch.Tensor):
      value = value.detach().cpu()
      if value.numel() == 1:
        return float(value.item())
      return value.tolist()
    if isinstance(value, dict):
      return {k: self._tensor_to_python(v) for k, v in value.items()}
    return value

  @torch.inference_mode()
  def evaluate_loss(self, dataloader) -> Dict[str, torch.Tensor]:
    assert (
      dataloader is not None
    ), "dataloader must be provided for evaluate_loss"
    self.model.eval()

    total_samples = torch.tensor(0.0, device=self.dist.env.device)
    loss_sums: Dict[str, torch.Tensor] = {}

    loss_forward_func = partial(self.arch, mode="loss")
    for batch_info in dataloader:
      batch_info = self.dist.prepare_for_distributed(batch_info)
      batch_info = DataSample(**batch_info).to(self.dist.env.device)
      batch_data, batch_meta, batch_latency = self.extract_batch_info(batch_info)
      self.arch.set_extra_attributes(meta=batch_meta, latency=batch_latency)
      with torch.autocast(device_type=self.dist.env.accelerator,
                          enabled=self.gradient.amp, dtype=self.dist.mix_dtype):
        _, loss_dict = loss_forward_func(batch_data, batch_info)

      labels = getattr(batch_info, "label", None)
      if labels is None:
        batch_size = len(batch_data)
      else:
        batch_size = int(torch.as_tensor(labels).shape[0])
      batch_size_tensor = torch.tensor(float(batch_size), device=self.dist.env.device)
      total_samples += batch_size_tensor

      total_loss = torch.zeros((), device=self.dist.env.device, dtype=torch.float32)
      for name, value in loss_dict.items():
        loss_value = value.detach().to(device=self.dist.env.device, dtype=torch.float32)
        total_loss += loss_value
        if name not in loss_sums:
          loss_sums[name] = torch.zeros_like(loss_value)
        loss_sums[name] += loss_value * batch_size_tensor
      if "loss" not in loss_sums:
        loss_sums["loss"] = torch.zeros_like(total_loss)
      loss_sums["loss"] += total_loss * batch_size_tensor

    if self.dist.env.is_dist_initialized():
      total_samples = self.dist.env.sync(total_samples, tensor_op="sum")
      if len(loss_sums) > 0:
        loss_sums = self.dist.env.sync(loss_sums, tensor_op="sum")

    if total_samples.item() <= 0:
      return {}

    loss_metrics = {key: value / total_samples for key, value in loss_sums.items()}
    return loss_metrics
  
  @torch.inference_mode()
  def evaluate(self, dataloader, result_file, 
                     phase: str = "test", 
                     step: Optional[int] = None
    ) -> Dict[str, Any]:
    """Evaluate on result_file, if not exists, infer first with dataloader."""
    if not Path(result_file).exists():
      results = self.infer(dataloader, result_file)
    else:
      results = mmengine.load(result_file)

    metrics = {}
    loss_metrics = self.evaluate_loss(dataloader)

    if self.dist.env.is_main_process():
      help_utils.print_log("Evaluating...")
      batch_info = results["batch_info"]
      batch_info = self.dist.prepare_for_distributed(batch_info)
      # for better performance, move to arch device first
      merge_result = DataSample.merge_from_list(batch_info).to(self.dist.env.device)
      output = merge_result.output # type: ignore
      targets = {key: getattr(merge_result, key) for key in dataloader.dataset.ann_keys}
      metrics = self.metric(output, targets)
      metrics.update(loss_metrics)

      # split figure and metric
      msg_list = []
      keys = list(metrics.keys())
      for key in keys:
        value = metrics[key]
        if isinstance(value, matplotlib.figure.Figure): # type: ignore
          value = metrics.pop(key)
          fig = value
          fig.savefig(str(result_file).replace(".pkl", f"_{key}.png"))
          continue

        assert (
          isinstance(value, torch.Tensor)
        ), "value must be a tensor"
        is_float = (not isinstance(value, torch.IntTensor))
        if is_float:
          value = value.round(decimals=6)

        if isinstance(value, torch.Tensor) and value.ndim > 0:
          msg_list.append(f"{key}: {'[' + ', '.join([(f'{v}') for v in value]) + ']'}")
        else:
          msg_list.append(f"{key}: {value}")

      help_utils.print_log(f"Metrics: {', '.join(msg_list)}")
      results["metrics"] = DataSample(**metrics).to("cpu").to_dict() # move to cpu to dump
      help_utils.print_log(f"Dump metric into: {result_file}")
      mmengine.dump(results, result_file)

      # visualize
      self.record.log_batch_sample(batch_info, output, targets, 
                                   class_names=dataloader.dataset.class_names,
                                   phase=phase, 
                                   progress_bar=True, step=step)

      if avg_loss_value := metrics.get("loss", None):
        metrics["avg_loss"] = avg_loss_value

      metrics, _ = self.record.format(metrics) # format data to cpu for further broadcast
      self.record.log_step_data(metrics, phase=phase, step=step)

    metrics = self.dist.env.broadcast(metrics)
    
    return metrics

  def evaluate_by_strategy(self, dataloader, phase: str = "test", 
                           epoch: int = -1, step: int = -1):
    if dataloader is None:
      return None

    def is_positive(x):
      return (x is not None and x > 0)

    eval_by_epoch, eval_by_step = False, False
    if is_positive(self.eval_every_n_epochs) and is_positive(epoch) and (epoch % self.eval_every_n_epochs == 0):
      eval_by_epoch = True
      result_file = self.work_dir / "results" / f"epoch_{epoch}.pkl"
    elif is_positive(self.eval_every_n_steps) and is_positive(step) and (step % self.eval_every_n_steps == 0):
      eval_by_step = True
      result_file = self.work_dir / "results" / f"step_{step}.pkl"

    if not eval_by_epoch and not eval_by_step:
      return

    if eval_by_epoch:
      eval_phase, index = f"{phase}_by_epoch", epoch - 1
    elif eval_by_step:
      eval_phase, index = f"{phase}_by_step", step - 1

    metrics = self.evaluate(dataloader, result_file, phase=eval_phase, step=index)

    if not eval_by_epoch:
      return metrics

    metric_plain = self._tensor_to_python(metrics)
    valid_loss = metric_plain.get("loss", None)
    ckpt_path = getattr(self, "last_epoch_ckpt_path", None)
    if isinstance(ckpt_path, Path):
      ckpt_path = ckpt_path.resolve()

    if isinstance(valid_loss, (int, float)):
      if valid_loss <= self.best_valid_loss:
        self.best_valid_loss = float(valid_loss)
        self.best_valid_epoch = int(epoch)
        if self.dist.env.is_main_process() and isinstance(ckpt_path, Path) and ckpt_path.exists():
          self.best_valid_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
          shutil.copy2(ckpt_path, self.best_valid_ckpt_path)
          help_utils.print_log(f"Update best valid checkpoint: {self.best_valid_ckpt_path}")

    epoch_record = dict(
      epoch=int(epoch),
      valid_loss=(float(valid_loss) if isinstance(valid_loss, (int, float)) else None),
      metrics=metric_plain,
      checkpoint=(str(ckpt_path) if ckpt_path is not None else None),
      result_file=str(result_file),
    )
    self.epoch_records.append(epoch_record)

    if self.dist.env.is_main_process():
      self.reporter.write_epoch_report(
        epoch=int(epoch),
        phase=eval_phase,
        metrics=metric_plain,
        result_file=Path(result_file),
        checkpoint_file=ckpt_path if isinstance(ckpt_path, Path) else None,
        best_valid_epoch=self.best_valid_epoch,
      )
    self.dist.env.sync()
    return metrics

  def evaluate_test_with_best_valid(self):
    if self.test_dataloader is None:
      return None
    best_ckpt = None
    if self.best_valid_ckpt_path.exists():
      best_ckpt = self.best_valid_ckpt_path
    elif self.best_train_ckpt_path.exists():
      best_ckpt = self.best_train_ckpt_path

    if best_ckpt is None:
      help_utils.print_log("No best checkpoint found, skip final test evaluation", level="WARNING")
      return None

    help_utils.print_log(f"Load best checkpoint for test: {best_ckpt}")
    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    model = self.dist.get_summon_module(self.model)
    model.load_state_dict(ckpt["weight"], strict=True)
    self.dist.env.sync()

    best_epoch = self.best_valid_epoch
    if best_epoch is None:
      best_epoch = ckpt.get("epoch", "unknown")
    test_result_file = self.work_dir / "results" / f"test_best_epoch_{best_epoch}.pkl"
    test_metrics = self.evaluate(self.test_dataloader, test_result_file, phase="test_best", step=None)
    test_metrics_plain = self._tensor_to_python(test_metrics)
    external_result_files = {}
    external_test_metrics = {}
    for external_name, external_loader in getattr(self, "external_test_dataloaders", {}).items():
      if external_loader is None:
        continue
      external_result_file = (
        self.work_dir / "results" / f"external_{external_name}_best_epoch_{best_epoch}.pkl"
      )
      external_metrics = self.evaluate(
        external_loader,
        external_result_file,
        phase=f"external_{external_name}_best",
        step=None,
      )
      external_result_files[external_name] = external_result_file
      external_test_metrics[external_name] = self._tensor_to_python(external_metrics)

    if self.dist.env.is_main_process():
      self.reporter.write_final_report(
        epoch_records=self.epoch_records,
        best_valid_epoch=self.best_valid_epoch,
        best_valid_loss=(None if self.best_valid_epoch is None else self.best_valid_loss),
        best_valid_ckpt=best_ckpt,
        test_result_file=test_result_file,
        test_metrics=test_metrics_plain,
        external_result_files=external_result_files,
        external_test_metrics=external_test_metrics,
      )
    self.dist.env.sync()
    return test_metrics

  def archive_env_for_reproducibility(self):
    """Archive config and source code under work dir for reproducibility."""

    # dump config
    try:
      dst_config_file = self.work_dir / "config.py"
      mmengine.Config.dump(self.cfg, dst_config_file)
    except Exception as e:
      help_utils.print_log(f"Error dumping config: {e}", level="ERROR")
      help_utils.print_log(f"Please check the config file", level="ERROR")
      exit(2)

    help_utils.print_log(f"Parsed Config: \n{self.cfg.dump()}")
    help_utils.print_log(f"Dumped config to: {dst_config_file}")

    # archive source code (exclude experiments so we do not copy work_dir into itself)
    try:
      source_code_dir = slimai.get_package_path()
      dst_source_code_dir = self.work_dir / "code"
      shutil.copytree(source_code_dir, dst_source_code_dir,
                      dirs_exist_ok=True,
                      ignore=shutil.ignore_patterns(
                        "*.pyc", ".git*", "._*", "_debug_", "experiments", "swanlog"
                      ))
    except Exception as e:
      help_utils.print_log(f"Error archiving source code: {e}", level="ERROR")
      exit(3)
      
    help_utils.print_log(f"Archived source code to: {dst_source_code_dir}")
    help_utils.print_log(f"Last git commit id: {slimai.get_last_commit_id()}")

    help_utils.print_log(f"Experiment work dir: {self.work_dir}")
    return
