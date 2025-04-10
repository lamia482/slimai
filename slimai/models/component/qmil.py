import torch
from slimai.helper.help_build import MODELS


__all__ = [
  "QMIL", 
]

@MODELS.register_module()
class QMIL(torch.nn.Module):

  MLP = MODELS.get("MLP")

  def __init__(self, *, input_dim, num_heads, num_layers, dropout=0.1, 
               act="gelu", norm="batch_norm", **kwargs):
    super().__init__()
    self.input_dim = input_dim
    self.cls_token = torch.nn.Parameter(torch.randn(1, input_dim))
    self.ln = torch.nn.LayerNorm(input_dim)
    self.atten_weight = self.MLP(input_dim=input_dim, output_dim=1, 
                                 hidden_dim=input_dim // num_heads, 
                                 bottleneck_dim=input_dim // num_heads, 
                                 n_layer=num_layers, act=act, norm=norm, dropout=dropout)
    self.proj = self.MLP(input_dim=input_dim*2, output_dim=input_dim, 
                        hidden_dim=input_dim, bottleneck_dim=input_dim, 
                        n_layer=num_layers, act=act, norm=norm, dropout=dropout)
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
      _neg_indices = torch.where(_weight < 0.5)[0]
      _pos_indices = torch.where(_weight >= 0.5)[0]
      _neg_weight, _neg_embeddings = _weight[_neg_indices], _x[_neg_indices]
      _pos_weight, _pos_embeddings = _weight[_pos_indices], _x[_pos_indices]

      _neg_weight = torch.pow(_neg_weight.unsqueeze(-1), 3/2) # < 0.5, 0.5 -> 0.35
      _pos_weight = torch.pow(_pos_weight.unsqueeze(-1), 2/3) # >= 0.5, 0.5 -> 0.62

      _neg_embeddings = (_neg_embeddings * _neg_weight) # [neg, D]
      _pos_embeddings = (_pos_embeddings * _pos_weight) # [pos, D]

      _neg_atten = self.get_atten(_neg_embeddings)
      _pos_atten = self.get_atten(_pos_embeddings)
      _embedding = torch.cat([_neg_atten, _pos_atten], dim=0) # [2D]

      batch_embedding.append(_embedding)

    batch_embedding = torch.stack(batch_embedding) # [B, 2D]
    projection = self.proj(batch_embedding) # [B, D]
    return projection

  def get_atten(self, ND):
    cls_token = self.cls_token.expand(1, -1)
    cls_embedding = torch.cat([cls_token, ND], dim=0).unsqueeze(0) # [1, 1+N, D]
    _z = self.ln(cls_embedding)
    _atten = _z
    for atten_layer in self.atten_topk:
      _atten, _ = atten_layer(_atten, _atten, _atten, need_weights=False)
      _atten = self.ln(_z + _atten)
    _atten = self.dropout(_atten)
    
    projection = _atten.squeeze(0)[0] # use CLS token as embedding
    return projection