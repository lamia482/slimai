import numpy as np
import torch
from typing import Dict
from slimai.helper.help_build import MODELS
from slimai.helper.help_utils import dist_env


@MODELS.register_module()
class DINOLoss(torch.nn.Module):
  def __init__(self, 
               output_dim, 
               warmup_teacher_temp, warmup_teacher_temp_epochs, 
               teacher_temp, student_temp=0.1, center_momentum=0.9
               ):
    super().__init__()
    self.student_temp = student_temp
    self.teacher_temp = teacher_temp
    self.center_momentum = center_momentum
    self.register_buffer("center", torch.zeros(1, output_dim))
    self.warmup_teacher_temp_schedule = np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs)
    return
  
  def forward(self, epoch, 
              student_n_crops, 
              teacher_n_crops, 
              student_output: torch.Tensor, 
              teacher_output: torch.Tensor) -> Dict[str, torch.Tensor]:
    # Compute classification loss
    if epoch >= len(self.warmup_teacher_temp_schedule):
      teacher_temp = self.teacher_temp
    else:
      teacher_temp = self.warmup_teacher_temp_schedule[epoch]

    student_out = torch.log_softmax(student_output / self.student_temp, dim=-1)
    student_out = student_out.chunk(teacher_n_crops + student_n_crops)

    teacher_out = torch.softmax((teacher_output - self.center) / teacher_temp, dim=-1)
    teacher_out = teacher_out.detach().chunk(teacher_n_crops)

    cls_loss = 0
    for i in range(teacher_n_crops + student_n_crops):
      for j in range(teacher_n_crops):
        # skip when view i and view j are the same
        if i == j:
          continue
        cls_loss += torch.sum(-teacher_out[j] * student_out[i], dim=-1).mean()
    cls_loss = cls_loss / student_n_crops / teacher_n_crops

    self.update_center(teacher_output)
    return dict(cls_loss=cls_loss)

  @torch.inference_mode()
  def update_center(self, teacher_output):
    """
    Update center used for teacher output.
    """
    batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
    batch_center = dist_env.sync(batch_center) # already divided by world size, no need to divide again

    # ema update
    self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
    return