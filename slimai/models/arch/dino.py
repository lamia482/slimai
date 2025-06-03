import torch
from typing import Union, Dict, List
from slimai.helper import help_utils
from slimai.helper.help_build import MODELS
from slimai.helper.structure import DataSample
from slimai.helper.utils import PytorchNetworkUtils
from .cls_arch import ClassificationArch
from ..component.pipeline import Pipeline


__all__ = [
  "DINO",
]

@MODELS.register_module()
class DINO(ClassificationArch):
  def __init__(self, *, 
               encoder=dict(
                 backbone=None, neck=None, 
               ), 
               decoder=dict(
                 head=None, 
               ), 
               loss=None, 
               solver=None, 
               momentum_teacher=0.9995, 
               ):
    super().__init__(encoder=encoder, decoder=decoder, loss=loss, solver=solver)
    
    # in DINO teacher use student weight by default
    self.teacher.load_state_dict(self.student.state_dict())
    
    # freeze teacher and only train student, ema student to teacher
    PytorchNetworkUtils.freeze(self.teacher)

    assert (
      0 < momentum_teacher < 1
    ), f"momentum_teacher must be in the range (0, 1), but got {momentum_teacher}"
    self.momentum_teacher = momentum_teacher
    self.momentum_teacher_schedule = None # update in step_precede_hooks
    return
  
  def init_layers(self, encoder, decoder) -> torch.nn.ModuleDict:
    student = Pipeline(encoder.backbone, encoder.neck, decoder.head)
    teacher = Pipeline(encoder.backbone, encoder.neck, decoder.head)    
    return torch.nn.ModuleDict(dict(teacher=teacher, student=student))
  
  @property
  def teacher(self) -> torch.nn.Module:
    return self.model.teacher # type: ignore
  
  @property
  def student(self) -> torch.nn.Module:
    return self.model.student # type: ignore
  
  def epoch_precede_hooks(self, *, runner):
    super().epoch_precede_hooks(runner=runner)
    self.student.train()
    self.teacher.eval()
    return
  
  def epoch_succeed_hooks(self, *, runner):
    super().epoch_succeed_hooks(runner=runner)
    self.student.eval()
    self.teacher.eval()
    return
  
  def step_precede_hooks(self, *, runner):
    super().step_precede_hooks(runner=runner)
    if self.momentum_teacher_schedule is None:
      self.momentum_teacher_schedule = help_utils.cosine_scheduler(
        base_value=self.momentum_teacher, final_value=1, epochs=self.max_train_epoch, 
        niter_per_ep=self.max_train_step, warmup_epochs=0, start_warmup_value=0
      )
    return

  def step_succeed_hooks(self, *, runner):
    # EMA update for the teacher
    assert (
      self.momentum_teacher_schedule is not None
    ), "momentum_teacher_schedule must be not None"
    with torch.inference_mode():
      global_train_step = self.current_train_epoch * self.max_train_step + self.current_train_step
      m = self.momentum_teacher_schedule[global_train_step]  # momentum parameter
      for ps, pt in zip(PytorchNetworkUtils.get_module_params(self.student), 
                        PytorchNetworkUtils.get_module_params(self.teacher)):
        pt.data.mul_(m).add_((1 - m) * ps.detach().data)
    return

  def _forward_tensor(self, 
                batch_data: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                return_flow: bool = False) -> Union[torch.Tensor, Dict[str, Union[int, torch.Tensor]]]:
    assert (
      "global_views" in batch_data and "local_views" in batch_data
    ), "batch_data must contain 'global_views' and 'local_views'"

    global_views_list: List[torch.Tensor] = batch_data["global_views"] # type: ignore
    local_views_list: List[torch.Tensor] = batch_data["local_views"] # type: ignore
    n_global_views, n_local_views = len(global_views_list), len(local_views_list)
    global_views, local_views = torch.cat(global_views_list), torch.cat(local_views_list)

    teacher_output = self.teacher(global_views)

    if return_flow:
      student_output = torch.cat([
        self.student(global_views), 
        self.student(local_views)
      ])
      return dict(
        student_n_crops=n_local_views, 
        student_output=student_output,
        teacher_n_crops=n_global_views, 
        teacher_output=teacher_output,
      )
    else:
      return teacher_output

  def _forward_loss(self, 
              embedding_dict: Dict[str, torch.Tensor], 
              batch_info: DataSample) -> Dict[str, torch.Tensor]:
    loss = self.loss(epoch=self.current_train_epoch, **embedding_dict)
    return loss
  
  def export_model(self) -> torch.nn.Module:
    # Export model for inference and export to onnx
    teacher_without_ddp = self.dist.get_summon_module(self.teacher)
    backbone = teacher_without_ddp.backbone
    return backbone # type: ignore
  
  def postprocess(self, 
                  batch_data: torch.Tensor, 
                  batch_info: DataSample) -> DataSample:
    embedding = batch_data
    batch_info.output = dict(
      embedding=embedding, # [B, D]
    )
    return batch_info
