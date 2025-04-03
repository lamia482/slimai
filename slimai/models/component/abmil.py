import torch
from slimai.helper.help_build import MODELS


__all__ = [
  "ABMIL", 
]

@MODELS.register_module()
class ABMIL(torch.nn.Module):
  def __init__(self, *, input_dim, num_heads, num_layers, dropout=0.1, 
               topk=16, 
               **kwargs):
    super().__init__()
    self.input_dim = input_dim
    self.cls_token = torch.nn.Parameter(torch.randn(1, 1, input_dim))
    self.ln = torch.nn.LayerNorm(input_dim)
    self.topk = topk
    self.atten_weight = torch.nn.Sequential(
      torch.nn.LayerNorm(input_dim), 
      torch.nn.Linear(input_dim, input_dim // num_heads), 
      torch.nn.GELU(), 
      torch.nn.Linear(input_dim // num_heads, 1), 
    )
    self.atten_topk = torch.nn.ModuleList([
      torch.nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
      for _ in range(num_layers)
    ])
    self.dropout = torch.nn.Dropout(dropout)
    return
  
  def forward(self, x):
    batch_embedding = []

    for _x in x:
      _bx = _x.unsqueeze(0) # [1, ~N, D]
      _weight = self.atten_weight(_bx).squeeze([0, 2]).sigmoid() # [~N]
      _sorted_indices = torch.argsort(_weight, descending=True)

      if self.topk <= 0:
        _topk_indices = _sorted_indices
      elif self.topk > len(_sorted_indices):
        _topk_indices = torch.hstack([
          _sorted_indices, 
          torch.repeat_interleave(_sorted_indices[-1], self.topk - len(_sorted_indices))
        ])
      else:
        _topk_indices = _sorted_indices[:self.topk]

      _topk_embedding = _x[_topk_indices] * _weight[_topk_indices].unsqueeze(-1) # [topk, D]
      batch_embedding.append(_topk_embedding)

    batch_cls_token = self.cls_token.expand(len(batch_embedding), -1, -1)
    batch_cls_embedding = []
    for cls_token, embedding in zip(batch_cls_token, batch_embedding):
      batch_cls_embedding.append(torch.cat([cls_token, embedding], dim=0))
      
    batch_cls_embedding = torch.stack(batch_cls_embedding) # [B, 1+topk, D]
    _z = self.ln(batch_cls_embedding)
    _atten = _z
    for atten_layer in self.atten_topk:
      _atten, _ = atten_layer(_atten, _atten, _atten, need_weights=False)
      _atten = self.ln(_z + _atten)
    _atten = self.dropout(_atten)
    
    final_embedding = _atten[:, 0, :] # use CLS token as embedding
    return final_embedding
