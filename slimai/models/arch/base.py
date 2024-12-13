from mmengine.model import BaseModel as MMengineBaseModel


class BaseModel(MMengineBaseModel):
  def __init__(self, 
               data_preprocessor, 
               backbone, 
               neck, 
               head):
    super().__init__()
    return
  def forward(self, batch_inputs, batch_data_samples, mode="_forward_"):
    expected_modes = ["_forward_", "_backward_", "_predict_"]
    if mode == "_forward_":
      return self._forward(batch_inputs, batch_data_samples)
    elif mode == "_predict_":
      return self._predict(batch_inputs, batch_data_samples)
    elif mode == "_backward_":
      return self._backward(batch_inputs, batch_data_samples)
    else:
      raise RuntimeError(f"Invalid mode \"{mode}\". Only supports {expected_modes}")

  def _forward_(self, batch_inputs, batch_data_samples):
    pass

  def _backward_(self, batch_inputs, batch_data_samples):
    pass

  def _predict_(self, batch_inputs, batch_data_samples):
    pass
