import torch
from torchmetrics.classification import MulticlassCohenKappa
from slimai.helper.help_build import MODELS
from slimai.helper.structure import DataSample
from .base_arch import BaseArch


@MODELS.register_module()
class ClassificationArch(BaseArch):
  def do_loss(self, 
              batch_data: torch.Tensor, 
              batch_info: DataSample):
    output = self.do_tensor(batch_data, batch_info)
    loss = self.loss(output, batch_info.label)
    kappa = MulticlassCohenKappa(num_classes=15).to(self.device)(
      batch_info.label, output.argmax(dim=1)
    )
    return dict(loss=loss, kappa=kappa)
  
  def postprocess(self, 
                  batch_data: torch.Tensor, 
                  batch_info: DataSample):
    return
