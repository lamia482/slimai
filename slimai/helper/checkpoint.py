import sys
import torch
from pathlib import Path
import mmengine
from typing import Optional, Union
from .help_utils import print_log
from .distributed import Distributed


__all__ = ["Checkpoint"]


class Checkpoint(object):
  """A class to handle model checkpoint saving and loading.
  
  This class provides functionality to save and load model checkpoints with various strategies
  including saving best, latest, and periodic checkpoints.
  
  Attributes:
    save_dir (Path): Directory to save checkpoints
    save_every_n_steps (int): Save checkpoint every N steps
    save_every_n_epochs (int): Save checkpoint every N epochs
    keep_max (int): Maximum number of checkpoints to keep
    keep_best (bool): Whether to keep the best checkpoint
    keep_latest (bool): Whether to keep the latest checkpoint
    record_file (Path): File to store checkpoint records
    min_loss (float): Minimum loss value for best checkpoint
    best_path (Path): Path to best checkpoint
    latest_path (Path): Path to latest checkpoint
  """
  
  def __init__(
    self,
    save_dir: str,
    save_every_n_steps: Optional[int] = None,
    save_every_n_epochs: int = 1,
    keep_max: int = -1,
    keep_best: bool = True,
    keep_latest: bool = True, 
    save_on_rank_0: bool = True,
  ):
    """Initialize Checkpoint handler.
    
    Args:
      work_dir: Working directory for saving checkpoints
      cfg: Configuration dictionary
      save_dir: Directory name to save checkpoints (default: "ckpts")
      save_every_n_epochs: Save checkpoint every N epochs (default: 1)
      keep_max: Maximum number of checkpoints to keep (default: -1, keep all)
      keep_best: Whether to keep the best checkpoint (default: True)
      keep_latest: Whether to keep the latest checkpoint (default: True)
    """
    self.dist = Distributed()

    self.save_dir = Path(save_dir).resolve()
    self.save_every_n_steps = save_every_n_steps
    self.save_every_n_epochs = save_every_n_epochs
    self.keep_max = keep_max
    self.keep_best = keep_best
    self.keep_latest = keep_latest
    self.record_file = self.save_dir / "stat.pkl"
    self.min_loss = float("inf")  # will be updated in `load`
    self.best_path = self.save_dir / "best.pth"
    self.latest_path = self.save_dir / "latest.pth"
    self.save_on_rank_0 = save_on_rank_0
    
    # Create checkpoint directory if it doesn't exist
    self.save_dir.mkdir(parents=True, exist_ok=True)
    return

  def __repr__(self):
    return (f"Checkpoint(\n"
            f"  save_dir={self.save_dir},\n"
            f"  save_every_n_steps={self.save_every_n_steps},\n"
            f"  save_every_n_epochs={self.save_every_n_epochs},\n"
            f"  keep_max={self.keep_max},\n"
            f"  keep_best={self.keep_best},\n"
            f"  keep_latest={self.keep_latest},\n"
            f"  save_on_rank_0={self.save_on_rank_0},\n"
            f")")
  __str__ = __repr__

  def save(
    self,
    model: torch.nn.Module,
    solver: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    step: int, 
    num_steps_per_epoch: int,
    loss: Optional[float] = None,
    export: bool = False, 
    **kwargs
  ) -> Path:
    """Save checkpoint with strategy in no ddp mode.
    
    Args:
      model: PyTorch model to save
      solver: Optimizer to save
      scheduler: Learning rate scheduler to save
      epoch: Current epoch number
      loss: Current loss value
      export: Whether to export model (default: False)
      **kwargs: Additional arguments
      
    Returns:
      Path to saved checkpoint
    """
    # step starts from 0, epoch starts from 1
    save_by_step = self.save_every_n_steps is not None and step >= 0 and step % self.save_every_n_steps == 0
    save_by_epoch = epoch >= 0 and epoch % self.save_every_n_epochs == 0
    if save_by_step:
      save_by_epoch = False # skip duplicate save by step and epoch

    if save_by_step:
      step_mod = step % num_steps_per_epoch
      epoch = step // num_steps_per_epoch
      ckpt_path = self.save_dir / f"step_{step}.pth" 
    else:
      step_mod = 0
      ckpt_path = self.save_dir / f"epoch_{epoch}.pth"

    if (
      (save_by_step or save_by_epoch)
      and (not self.save_on_rank_0 or self.dist.env.is_main_process())
    ):
      if not self.save_on_rank_0:
        ckpt_path = str(ckpt_path).replace(".pth", f"-rank_{self.dist.env.local_rank}.pth")
        ckpt_path = Path(ckpt_path)

      update_best = False
      if loss is not None and loss <= self.min_loss:
        self.min_loss = loss
        update_best = True and self.keep_best

      def _save(ckpt: Path, export: bool = False) -> None:
        Path(ckpt).resolve().parent.mkdir(parents=True, exist_ok=True)
        print_log(f"Save checkpoint to {ckpt}")
        summon_module = self.dist.get_summon_module(model)
        torch.save(dict(weight=self.dist.copy_cpu_offload_state_dict(summon_module),
                        solver=self.dist.copy_cpu_offload_state_dict(solver),
                        scheduler=self.dist.copy_cpu_offload_state_dict(scheduler),
                        step=step_mod, epoch=epoch, 
                        loss=loss, min_loss=self.min_loss, 
                        **kwargs), ckpt)
        
        if export:
          raise NotImplementedError("Export is not implemented yet")
        return
      
      _save(ckpt_path, export=export)

      if self.keep_latest:
        self.latest_path.unlink(missing_ok=True)
        self.latest_path.symlink_to(ckpt_path)

      if update_best:
        _save(self.best_path, export=export)

      if not self.record_file.exists():
        mmengine.dump([], self.record_file)

      records = mmengine.load(self.record_file)
      records.append(dict(
        ckpt=ckpt_path, epoch=epoch, 
        global_step=step, num_steps_per_epoch=num_steps_per_epoch, step=step_mod,
        loss=loss, min_loss=self.min_loss, **kwargs
      ))
      if (
        (self.keep_max is not None and self.keep_max > 0) 
        and (len(records) > self.keep_max)
      ):
        path = Path(records[-self.keep_max-1]["ckpt"]).resolve()
        path.unlink(missing_ok=True)

      mmengine.dump(records, self.record_file)

    if self.save_on_rank_0:
      self.dist.env.sync()

    return ckpt_path

  def load(
    self,
    model: Optional[torch.nn.Module] = None,
    solver: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    *,
    resume: bool = True,
    resume_from: Optional[Union[str, int, Path]] = None,
    load_from: Optional[Union[str, Path]] = None,
    strict: bool = True
  ):
    """Load or resume from checkpoint in no ddp mode.
    
    Args:
      model: PyTorch model to load weights into (optional)
      solver: Optimizer to load state into (optional)
      scheduler: Learning rate scheduler to load state into (optional)
      resume: Whether to resume training (default: True)
      resume_from: Checkpoint to resume from, can be "best", "latest", epoch number, or path
      load_from: Direct path to checkpoint to load from
      strict: Whether to strictly enforce that the keys in state_dict match the keys returned by this module's state_dict() function (default: True)
    Returns:
      Loaded model
    """
    if resume_from in ["best", "latest"]:
      resume_from = getattr(self, f"{resume_from}_path")
    elif isinstance(resume_from, int):
      resume_from = self.save_dir / f"epoch_{resume_from}.pth"

    resume_from = Path(str(resume_from)).resolve()
    
    # Only resume when checkpoint exists
    if resume_from is None or not Path(resume_from).exists():
      resume = False
    else:
      resume = True
      load_from = resume_from

    if load_from is None:
      ckpt = dict()
      print_log("No checkpoint to load, build from scratch", level="WARNING")
      assert (
        model is not None
      ), "model must be provided when no checkpoint to load"
    else:
      load_from = Path(load_from).resolve()
      if not load_from.exists():
        print_log(f"Checkpoint {load_from} not found", level="ERROR")
        sys.exit(2)

      print_log(f"{'Resume' if resume else 'Load'} checkpoint from {load_from}")
      ckpt = torch.load(load_from, map_location="cpu", weights_only=False)

      if model is None:
        if None in [solver, scheduler]:
          raise ValueError("model must be provided when solver or scheduler is not provided")
        from .help_build import build_model
        arch = build_model(ckpt["cfg"]) # build pure model in no ddp mode
        model, solver, scheduler, _ = arch.extract() # type: ignore
        
      assert (
        model is not None
      ), "model must be provided when no checkpoint to load"

      # model is expected to be in non distributed style and load weights
      model.load_state_dict(ckpt["weight"], strict=(resume or strict))
      
      if solver and (solver_state := ckpt.get("solver", None)):
        solver.load_state_dict(solver_state)
      if scheduler and (scheduler_state := ckpt.get("scheduler", None)):
        scheduler.load_state_dict(scheduler_state)

      if resume:
        min_loss = ckpt.get("min_loss", None)
        self.min_loss = min_loss or float("inf")
      else:
        ckpt.pop("step", None)
        ckpt.pop("epoch", None)
        ckpt.pop("min_loss", None)
        
    return model, solver, scheduler, ckpt
