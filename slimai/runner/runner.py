import os
import time
import copy
import json
import torch
import shutil
import matplotlib
from pathlib import Path
from functools import partial
from typing import Dict, Any, Optional, List, Tuple
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

    self.record = Record(cfg=cfg, 
                         no_record=cfg.RUNNER.logger.get("no_record", False))
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
    resolved_resume_from = self._resolve_resume_from(resume.resume_from) if resume.enable else None
    if resume.enable and isinstance(resolved_resume_from, str):
      help_utils.print_log(f"Resume checkpoint source: {resolved_resume_from}")
    model, solver, scheduler, ckpt = self.checkpoint.load(
      model=model,
      solver=solver,
      scheduler=scheduler,
      resume=resume.enable,
      resume_from=resolved_resume_from,
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
    self.dump_runtime_metadata()
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

      if "loss" in loss_dict:
        total_loss = loss_dict["loss"]
      else:
        total_loss = None
        loss_keys = list(loss_dict.keys())
        for loss_name in loss_keys:
          loss_value = loss_dict[loss_name]
          if "loss" in loss_name:
            if total_loss is None:
              total_loss = torch.zeros_like(loss_value)
            total_loss += loss_value
      assert total_loss is not None, (
        "loss dict must contain key 'loss' or at least one key including 'loss'."
      )

      # add '^' to loss names for visualization; do not rename main "loss" key
      renamed_loss_dict = {}
      for loss_name, loss_value in loss_dict.items():
        if "loss" in loss_name and loss_name != "loss":
          renamed_loss_dict[f"{loss_name}^"] = loss_value
        else:
          renamed_loss_dict[loss_name] = loss_value
      loss_dict = renamed_loss_dict
      
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
    self.evaluate_all_with_best_valid()
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

  def _resolve_best_valid_metric(self, metric_plain: Dict[str, Any]):
    metric_name = self.cfg.RUNNER.get("best_ckpt_metric", "loss")
    if metric_name in ["loss", "valid_loss"]:
      candidate = metric_plain.get("loss", None)
    else:
      candidate = metric_plain.get(metric_name, None)
      if candidate is None:
        candidate = metric_plain.get("loss", None)
    if isinstance(candidate, list):
      candidate = candidate[0] if len(candidate) > 0 else None
    return metric_name, candidate

  @staticmethod
  def _update_symlink(link_path: Path, target_path: Path) -> None:
    link_path = Path(link_path)
    target_path = Path(target_path).resolve()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
      link_path.unlink(missing_ok=True)
    try:
      rel_target = Path(os.path.relpath(str(target_path), start=str(link_path.parent)))
      link_path.symlink_to(rel_target)
    except Exception:
      link_path.symlink_to(target_path)
    return

  def _resolve_resume_from(self, resume_from: Any) -> Any:
    if resume_from is None:
      return None
    if isinstance(resume_from, int):
      return resume_from

    text = str(resume_from).strip()
    if text == "":
      return None

    if text in ["latest", "best"]:
      link_path = self.checkpoint.latest_path if text == "latest" else self.checkpoint.best_path
      if link_path.exists():
        resolved = link_path.resolve()
        help_utils.print_log(f"Resolved resume_from={text} -> {resolved}")
        return str(resolved)

      epoch_candidates: List[Tuple[int, Path]] = []
      for ckpt_path in self.checkpoint.save_dir.glob("epoch_*.pth"):
        stem = ckpt_path.stem
        epoch_text = stem.split("epoch_")[-1]
        if epoch_text.isdigit():
          epoch_candidates.append((int(epoch_text), ckpt_path.resolve()))
      if len(epoch_candidates) > 0:
        epoch_candidates = sorted(epoch_candidates, key=lambda kv: kv[0])
        fallback_path = epoch_candidates[-1][1]
        help_utils.print_log(
          f"resume_from={text} is unavailable, fallback to explicit checkpoint: {fallback_path}",
          level="WARNING",
        )
        return str(fallback_path)
      help_utils.print_log(
        f"resume_from={text} not found under {self.checkpoint.save_dir}, fallback to build from scratch",
        level="WARNING",
      )
      return text

    resume_path = Path(text).expanduser()
    if not resume_path.is_absolute():
      in_work_dir = (self.work_dir / resume_path)
      if in_work_dir.exists():
        resume_path = in_work_dir
      else:
        resume_path = (Path.cwd() / resume_path)
    if resume_path.exists():
      resolved = resume_path.resolve()
      help_utils.print_log(f"Resolved explicit resume checkpoint -> {resolved}")
      return str(resolved)
    return text

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

      if "loss" in loss_dict:
        total_loss = loss_dict["loss"].detach().to(device=self.dist.env.device, dtype=torch.float32)
      else:
        total_loss = None
      for name, value in loss_dict.items():
        loss_value = value.detach().to(device=self.dist.env.device, dtype=torch.float32)
        if ("loss" not in loss_dict) and ("loss" in name):
          if total_loss is None:
            total_loss = torch.zeros((), device=self.dist.env.device, dtype=torch.float32)
          total_loss += loss_value
        if name not in loss_sums:
          loss_sums[name] = torch.zeros_like(loss_value)
        loss_sums[name] += loss_value * batch_size_tensor
      if total_loss is None:
        total_loss = torch.zeros((), device=self.dist.env.device, dtype=torch.float32)
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
    best_metric_name, valid_loss = self._resolve_best_valid_metric(metric_plain)
    ckpt_path = getattr(self, "last_epoch_ckpt_path", None)
    if isinstance(ckpt_path, Path):
      ckpt_path = ckpt_path.resolve()

    if isinstance(valid_loss, (int, float)):
      if valid_loss <= self.best_valid_loss:
        self.best_valid_loss = float(valid_loss)
        self.best_valid_epoch = int(epoch)
        if self.dist.env.is_main_process() and isinstance(ckpt_path, Path) and ckpt_path.exists():
          self._update_symlink(self.best_valid_ckpt_path, ckpt_path)
          help_utils.print_log(
            f"Update best valid checkpoint by {best_metric_name}: {self.best_valid_ckpt_path}"
          )

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

  def _build_train_analysis_dataloader(self):
    if self.cfg.get("TRAIN_LOADER", None) is None:
      return None
    cfg_loader = copy.deepcopy(self.cfg.TRAIN_LOADER)
    cfg_loader["shuffle"] = False
    dataset_cfg = cfg_loader.get("dataset", {})
    if isinstance(dataset_cfg, dict):
      dataset_cfg["balance"] = False
      dataset_cfg["repeat"] = 1
      dataset_cfg["augmenter"] = None
      if "with_augment" in dataset_cfg:
        dataset_cfg["with_augment"] = False
      if subsample := self.cfg.get("TRAIN_EVAL_SUBSAMPLE", None):
        dataset_cfg["max_sample_num"] = subsample
    return help_build.build_dataloader(cfg_loader)

  def evaluate_all_with_best_valid(self):
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

    help_utils.print_log(f"Load best checkpoint for analysis: {best_ckpt}")
    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    model = self.dist.get_summon_module(self.model)
    model.load_state_dict(ckpt["weight"], strict=True)
    self.dist.env.sync()

    best_epoch = self.best_valid_epoch
    if best_epoch is None:
      best_epoch = ckpt.get("epoch", "unknown")
    train_analysis_loader = self._build_train_analysis_dataloader()
    analysis_result_files = {}
    analysis_metrics = {}

    if train_analysis_loader is not None:
      train_result_file = self.work_dir / "results" / f"analysis_train_best_epoch_{best_epoch}.pkl"
      train_metrics = self.evaluate(train_analysis_loader, train_result_file, phase="train_analysis_best", step=None)
      analysis_result_files["train"] = train_result_file
      analysis_metrics["train"] = self._tensor_to_python(train_metrics)

    if self.valid_dataloader is not None:
      valid_result_file = self.work_dir / "results" / f"analysis_valid_best_epoch_{best_epoch}.pkl"
      valid_metrics = self.evaluate(self.valid_dataloader, valid_result_file, phase="valid_analysis_best", step=None)
      analysis_result_files["valid"] = valid_result_file
      analysis_metrics["valid"] = self._tensor_to_python(valid_metrics)

    test_result_file = self.work_dir / "results" / f"analysis_test_best_epoch_{best_epoch}.pkl"
    test_metrics = self.evaluate(self.test_dataloader, test_result_file, phase="test_analysis_best", step=None)
    analysis_result_files["test"] = test_result_file
    analysis_metrics["test"] = self._tensor_to_python(test_metrics)

    test_metrics_plain = analysis_metrics["test"]
    external_result_files = {}
    external_test_metrics = {}
    for external_name, external_loader in getattr(self, "external_test_dataloaders", {}).items():
      if external_loader is None:
        continue
      external_result_file = (
        self.work_dir / "results" / f"analysis_external_{external_name}_best_epoch_{best_epoch}.pkl"
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
        analysis_result_files=analysis_result_files,
        analysis_metrics=analysis_metrics,
        test_result_file=test_result_file,
        test_metrics=test_metrics_plain,
        external_result_files=external_result_files,
        external_test_metrics=external_test_metrics,
      )
    self.dist.env.sync()
    return test_metrics

  def archive_env_for_reproducibility(self):
    """Archive config and source code under work dir for reproducibility."""

    is_global_main = self.dist.env.is_main_process(local=False)

    # dump config
    if is_global_main:
      try:
        dst_config_file = self.work_dir / "config.py"
        mmengine.Config.dump(self.cfg, dst_config_file)
      except Exception as e:
        help_utils.print_log(f"Error dumping config: {e}", level="ERROR")
        help_utils.print_log(f"Please check the config file", level="ERROR")
        exit(2)

      help_utils.print_log(f"Parsed Config: \n{self.cfg.dump()}")
      help_utils.print_log(f"Dumped config to: {dst_config_file}")

      # snapshot dataset files at the beginning of training
      try:
        self.snapshot_dataset_files()
      except Exception as e:
        help_utils.print_log(f"Failed to snapshot dataset files: {e}", level="WARNING")

      # dump taxonomy metadata
      try:
        taxonomy_file = self.work_dir / "label_taxonomy.json"
        taxonomy_payload = {}
        for key in [
          "LABEL_TAXONOMY",
          "PRIMARY_LABEL_MAPPING",
          "SECONDARY_LABEL_MAPPING",
          "SECONDARY_TO_PRIMARY",
          "SECONDARY_LOCAL_MAPPING",
          "LABEL_LEVELS",
          "TAXONOMY_VERSION",
          "TAXONOMY_HASH",
        ]:
          value = self.cfg.get(key, None)
          if value is not None:
            taxonomy_payload[key] = value
        taxonomy_file.write_text(json.dumps(taxonomy_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        help_utils.print_log(f"Dumped taxonomy metadata to: {taxonomy_file}")
      except Exception as e:
        help_utils.print_log(f"Failed to dump taxonomy metadata: {e}", level="WARNING")

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

    if self.dist.env.is_dist_initialized():
      self.dist.env.sync()

    help_utils.print_log(f"Experiment work dir: {self.work_dir}")
    return

  def _collect_dataset_snapshot_candidates(self) -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []

    def _append_candidate(tag: str, value: Any):
      if not isinstance(value, str):
        return
      text = value.strip()
      if text == "":
        return
      candidates.append((tag, text))
      return

    for key in [
      "EXCEL_FILE",
      "OUTPUT_SPLIT_FILE",
      "SPLIT_FILE",
      "DATASET_FILE",
      "ANNOTATION_FILE",
    ]:
      _append_candidate(key.lower(), self.cfg.get(key, None))

    external_excel_files = self.cfg.get("EXTERNAL_EXCEL_FILES", None)
    if isinstance(external_excel_files, dict):
      for name, path in external_excel_files.items():
        _append_candidate(f"external_excel_{name}", path)

    for key in ["BREXI_SOURCE", "SOURCE"]:
      source_cfg = self.cfg.get(key, None)
      if not isinstance(source_cfg, dict):
        continue
      _append_candidate(f"{key.lower()}_sheet_file", source_cfg.get("sheet_file", None))
      _append_candidate(f"{key.lower()}_output_split_file", source_cfg.get("output_split_file", None))

    return candidates

  def snapshot_dataset_files(self):
    dataset_dir = self.work_dir / "dataset"
    source_dir = dataset_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    records = []
    used_names = set()
    for tag, src_path_str in self._collect_dataset_snapshot_candidates():
      src_path = Path(src_path_str).expanduser()
      if not src_path.is_absolute():
        src_path = (Path.cwd() / src_path).resolve()
      else:
        src_path = src_path.resolve()

      record = dict(
        tag=tag,
        source_path=str(src_path),
        exists=bool(src_path.exists()),
        copied=False,
        snapshot_path="",
        size_bytes=None,
        error="",
      )

      if not src_path.exists() or not src_path.is_file():
        records.append(record)
        continue

      safe_tag = "".join(ch if (ch.isalnum() or ch in ["-", "_"]) else "_" for ch in tag)
      dst_name = f"{safe_tag}__{src_path.name}"
      if dst_name in used_names:
        dst_name = f"{safe_tag}__{len(used_names):04d}__{src_path.name}"
      used_names.add(dst_name)
      dst_path = source_dir / dst_name

      try:
        shutil.copy2(src_path, dst_path)
        record["copied"] = True
        record["snapshot_path"] = str(dst_path)
        record["size_bytes"] = int(dst_path.stat().st_size)
      except Exception as e:
        record["error"] = str(e)
      records.append(record)

    manifest_path = dataset_dir / "dataset_snapshot_manifest.json"
    payload = dict(
      schema_version="dataset_snapshot_manifest_v1",
      run_dir=str(self.work_dir),
      source_dir=str(source_dir),
      files=records,
    )
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    help_utils.print_log(f"Dumped dataset snapshot manifest to: {manifest_path}")
    return

  def dump_runtime_metadata(self):
    try:
      dataset_meta_file = self.work_dir / "dataset_meta.json"
      dataset_meta = {}
      for split_name, dataloader in [
        ("train", getattr(self, "train_dataloader", None)),
        ("valid", getattr(self, "valid_dataloader", None)),
        ("test", getattr(self, "test_dataloader", None)),
      ]:
        if dataloader is None:
          continue
        dataset = dataloader.dataset
        dataset_meta[split_name] = dict(
          split_file=getattr(dataset, "split_file", None),
          split_stat=getattr(dataset, "split_stat", None),
          split_diag=getattr(dataset, "split_diag", None),
          split_diag_file=getattr(dataset, "split_diag_file", None),
        )
      external_meta = {}
      for external_name, dataloader in getattr(self, "external_test_dataloaders", {}).items():
        if dataloader is None:
          continue
        dataset = dataloader.dataset
        external_meta[external_name] = dict(
          split_file=getattr(dataset, "split_file", None),
          split_stat=getattr(dataset, "split_stat", None),
          split_diag=getattr(dataset, "split_diag", None),
          split_diag_file=getattr(dataset, "split_diag_file", None),
        )
      dataset_meta["external"] = external_meta
      dataset_meta_file.write_text(json.dumps(dataset_meta, ensure_ascii=False, indent=2), encoding="utf-8")
      help_utils.print_log(f"Dumped runtime dataset metadata to: {dataset_meta_file}")
    except Exception as e:
      help_utils.print_log(f"Failed to dump runtime dataset metadata: {e}", level="WARNING")
    return
