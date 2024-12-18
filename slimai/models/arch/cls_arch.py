from .base import BaseArch, MODELS


@MODELS.register_module()
class ClassificationArch(BaseArch):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    return
  
  def _init_layers(self):
    pass
  
  def tensor(self, batch_inputs, batch_data_samples):
    pass

  def loss(self, batch_inputs, batch_data_samples):
    pass

  def predict(self, batch_inputs, batch_data_samples):
    pass
