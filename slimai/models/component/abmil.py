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
    self.atten_rough = torch.nn.ModuleList([
      torch.nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
      for _ in range(num_layers)
    ])
    self.atten_weight = torch.nn.Linear(input_dim, 1)
    self.atten_topk = torch.nn.ModuleList([
      torch.nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
      for _ in range(num_layers)
    ])
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
      
      _atten = _z
      for atten_layer in self.atten_rough:
        _atten, _ = atten_layer(_atten, _atten, _atten, need_weights=False)
        _atten = self.ln(_z + _atten) # [1, ~N, D]

      _weight = self.atten_weight(_atten).squeeze([0, 2]).softmax(dim=0) # [~N]
      _sorted_indices = torch.argsort(_weight, descending=True)
      
      if self.topk > len(_sorted_indices):
        _topk_indices = torch.hstack([
          _sorted_indices, 
          torch.repeat_interleave(_sorted_indices[-1], self.topk - len(_sorted_indices))
        ])
      else:
        _topk_indices = _sorted_indices[:self.topk]
      
      _topk_embedding = _y[_topk_indices] # [topk, D]
      batch_embedding.append(_topk_embedding)
      
    batch_embedding = torch.stack(batch_embedding) # [B, topk, D]
    _z = self.ln(batch_embedding)
    _atten = _z
    for atten_layer in self.atten_topk:
      _atten, _ = atten_layer(_atten, _atten, _atten, need_weights=False)
      _atten = self.ln(_z + _atten)
    _atten = self.dropout(_atten)
    
    final_embedding = _atten[:, 0, :] # use CLS token as embedding
    return final_embedding
