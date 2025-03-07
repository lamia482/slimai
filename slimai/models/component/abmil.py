import torch
from slimai.helper.help_build import MODELS


__all__ = [
  "ABMIL", 
]

@MODELS.register_module()
class ABMIL(torch.nn.Module):
  def __init__(self, *, input_dim, num_heads, dropout=0.1, 
               **kwargs):
    super().__init__()
    self.input_dim = input_dim
    self.cls_token = torch.nn.Parameter(torch.randn(1, 1, input_dim))
    self.ln = torch.nn.LayerNorm(input_dim)
    self.atten = torch.nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
    self.dropout = torch.nn.Dropout(dropout)
    return
  
  def forward(self, x):
    batch_cls_token = self.cls_token.expand(len(x), -1, -1)
    y = []
    for cls_token, _x in zip(batch_cls_token, x):
      y.append(torch.cat([cls_token, _x], dim=0))

    batch_embedding = []
    for _y in y:
      _z = _y.unsqueeze(0) # [1, ~N, D]
      _z = self.ln(_z)
      _atten, _ = self.atten(_z, _z, _z, need_weights=False)
      _atten = self.dropout(_atten)
      _embedding = _atten.squeeze(0)[0] # use CLS token as embedding
      batch_embedding.append(_embedding)
    
    batch_embedding = torch.stack(batch_embedding)
    return batch_embedding
