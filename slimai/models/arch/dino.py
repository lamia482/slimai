import torch
from typing import Union, Dict
from slimai.helper import help_utils
from slimai.helper.help_build import MODELS
from slimai.helper.structure import DataSample
from .base_arch import BaseArch
from ..component.pipeline import Pipeline


__all__ = [
  "DINO",
]

@MODELS.register_module()
class DINO(BaseArch):
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

    assert (
      0 < momentum_teacher < 1
    ), f"momentum_teacher must be in the range (0, 1), but got {momentum_teacher}"
    self.momentum_teacher = momentum_teacher
    self.momentum_teacher_schedule = None # update in step_precede_hooks
    return
  
  def init_layers(self, encoder, decoder):
    student = Pipeline(encoder.backbone, encoder.neck, decoder.head)
    teacher = Pipeline(encoder.backbone, encoder.neck, decoder.head)    
    return torch.nn.ModuleDict(dict(teacher=teacher, student=student))
  
  @property
  def teacher(self):
    return self.model.teacher
  
  @property
  def student(self):
    return self.model.student
  
  def epoch_precede_hooks(self, *, runner):
    super().epoch_precede_hooks(runner=runner)
    
    # freeze teacher and only train student, ema student to teacher
    help_utils.PytorchNetworkUtils.freeze(self.teacher)
    
    self.student.train()
    self.teacher.eval()
    return
  
  def epoch_succeed_hooks(self, *, runner):
    super().epoch_succeed_hooks(runner=runner)
    return
  
  def step_precede_hooks(self, *, runner):
    super().step_precede_hooks(runner=runner)
    if self.momentum_teacher_schedule is None:
      self.momentum_teacher_schedule = help_utils.cosine_scheduler(
        base_value=self.momentum_teacher, final_value=1, epochs=self.max_train_epoch, 
        niter_per_ep=self.max_train_step, warmup_epochs=1, start_warmup_value=0
      )
    return

  def step_succeed_hooks(self, *, runner):
    # EMA update for the teacher
    with torch.no_grad():
      global_train_step = self.current_train_epoch * self.max_train_step + self.current_train_step
      m = self.momentum_teacher_schedule[global_train_step]  # momentum parameter
      for ps, pt in zip(help_utils.PytorchNetworkUtils.get_module_params(self.student), 
                        help_utils.PytorchNetworkUtils.get_module_params(self.teacher)):
        pt.data.mul_(m).add_((1 - m) * ps.detach().data)
    return

  def _forward_tensor(self, 
                batch_data: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                return_flow: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    global_views, local_views = batch_data["global_views"], batch_data["local_views"]
    n_global_views, n_local_views = len(global_views), len(local_views)
    global_views, local_views = torch.cat(global_views), torch.cat(local_views)

    teacher_output = self.teacher(global_views)
    student_output = self.student(local_views)

    if return_flow:
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
    loss = self.loss(epoch=self.epoch, **embedding_dict)
    return loss

  def postprocess(self, 
                  batch_data: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                  batch_info: DataSample) -> DataSample:
    # Postprocess the output by assigning it to batch_info
    batch_info.output = batch_data
    return batch_info
