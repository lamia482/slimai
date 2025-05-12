import torch
from slimai.helper.help_build import MODELS


__all__ = [
  "RABMIL", "QMIL", 
]

@MODELS.register_module()
class RABMIL(torch.nn.Module):
  """
  Recursive Attention Based MIL
  compute attention(weight) for each patch, then recursively 
  re-compute attention for the topk-percent patches applied with attention weight 
  untill the final one embedding vector.
  """

  MLP = MODELS.get("MLP")

  def __init__(self, *, input_dim, num_heads, num_layers, dropout=0.1, topk_percent=0.5, 
               act="gelu", norm="batch_norm", kernel=None, **kwargs):
    super().__init__()
    self.input_dim = input_dim
    self.ln = torch.nn.LayerNorm(input_dim)
    # Attention weight network for calculating importance of each patch
    self.atten_weight = self.MLP(input_dim=input_dim, output_dim=1, 
                                 hidden_dim=input_dim // num_heads, 
                                 bottleneck_dim=input_dim // num_heads, 
                                 n_layer=num_layers, act=act, norm=norm, dropout=dropout)
    # Projection network for final feature transformation
    self.proj = self.MLP(input_dim=input_dim, output_dim=input_dim, 
                        hidden_dim=input_dim, bottleneck_dim=input_dim, 
                        n_layer=num_layers, act=act, norm=norm, dropout=dropout)
    # Multi-head attention layers for feature extraction
    self.atten_topk = torch.nn.ModuleList([
      torch.nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
      for _ in range(num_layers)
    ])
    self.dropout = torch.nn.Dropout(dropout)
    self.topk_percent = topk_percent  # Percentage of top-k patches to keep in each iteration
    self.kernel = kernel
    return
  
  def forward(self, x):
    batch_embedding = []

    # Process each sample in the batch individually
    for _x in x:
      _embedding = self.get_recursive_atten(_x)
      batch_embedding.append(_embedding)

    batch_embedding = torch.stack(batch_embedding) # [B, D]
    projection = self.proj(batch_embedding) # [B, D]
    return projection

  def get_recursive_atten(self, ND):
    # ND: [N, D] - N patches, each with dimension D
    while len(ND) > 1:
      # Calculate attention weight for each patch
      weight = self.get_atten_weight(ND) # [N]
      # Select top-k patches with highest weights
      n_topk = max(1, int(len(ND) * self.topk_percent))
      topk_weight, topk_indices = torch.topk(weight, n_topk, dim=0)

      # Apply weights to selected patches
      if self.kernel is None:
        ND = ND[topk_indices] * topk_weight.unsqueeze(-1) # [topk, D]
      elif self.kernel == "A":
        ND = ND[topk_indices] # [topk, D]

    ND = ND.squeeze(0) # [D]
    return ND
    
  def get_atten_weight(self, ND):
    embeddings = ND.unsqueeze(0) # [1, N, D]

    if self.kernel is None:
      _z = self.ln(embeddings)
      _atten = _z
      # Process features through multiple attention layers
      for atten_layer in self.atten_topk:
        _atten, _ = atten_layer(_atten, _atten, _atten, need_weights=False)
        _atten = self.ln(_z + _atten)
      _atten = self.dropout(_atten) # [1, N, D]

      # Calculate final attention weights
      weight = self.atten_weight(_atten).squeeze() # [N]
    elif self.kernel == "A":
      _z = self.ln(embeddings)
      _atten = _z
      # Process features through multiple attention layers
      for atten_layer in self.atten_topk:
        _atten, _ = atten_layer(_atten, _atten, _atten, need_weights=False)
        _atten = self.dropout(_atten)
        _atten = _z + _atten

      # Calculate final attention weights
      weight = self.atten_weight(_atten).squeeze().sigmoid() # [N]
    return weight


@MODELS.register_module()
class QMIL(torch.nn.Module):
  """
  Quadrant Multiple Instance Learning
  Divides instances into positive and negative parts, computes attention embeddings 
  separately, then merges them into a single feature vector
  """

  MLP = MODELS.get("MLP")

  def __init__(self, *, input_dim, num_heads, num_layers, dropout=0.1, 
               act="gelu", norm="batch_norm", **kwargs):
    super().__init__()
    self.input_dim = input_dim
    # Classification token for feature aggregation
    self.cls_token = torch.nn.Parameter(torch.randn(1, input_dim))
    self.ln = torch.nn.LayerNorm(input_dim)
    # Attention weight calculation network
    self.atten_weight = self.MLP(input_dim=input_dim, output_dim=1, 
                                 hidden_dim=input_dim // num_heads, 
                                 bottleneck_dim=input_dim // num_heads, 
                                 n_layer=num_layers, act=act, norm=norm, dropout=dropout)
    # Projection network to transform concatenated positive/negative features
    self.proj = self.MLP(input_dim=input_dim*2, output_dim=input_dim, 
                        hidden_dim=input_dim, bottleneck_dim=input_dim, 
                        n_layer=num_layers, act=act, norm=norm, dropout=dropout)
    # Multi-head attention layers
    self.atten_topk = torch.nn.ModuleList([
      torch.nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
      for _ in range(num_layers)
    ])
    self.dropout = torch.nn.Dropout(dropout)
    return
  
  def forward(self, x):
    batch_embedding = []

    # Process each sample in the batch individually
    for _x in x:
      _bx = _x.unsqueeze(0) # [1, ~N, D]
      # Calculate weight for each patch and apply sigmoid activation
      _weight = self.atten_weight(_bx).squeeze([0, 2]).sigmoid() # [~N]
      # Divide patches into negative (<0.5) and positive (>=0.5) samples
      _neg_indices = torch.where(_weight < 0.5)[0]
      _pos_indices = torch.where(_weight >= 0.5)[0]
      _neg_weight, _neg_embeddings = _weight[_neg_indices], _x[_neg_indices]
      _pos_weight, _pos_embeddings = _weight[_pos_indices], _x[_pos_indices]

      # Apply non-linear transformations to weights to enhance differences
      _neg_weight = torch.pow(_neg_weight.unsqueeze(-1), 3/2) # < 0.5, 0.5 -> 0.35
      _pos_weight = torch.pow(_pos_weight.unsqueeze(-1), 2/3) # >= 0.5, 0.5 -> 0.62

      # Apply weights to corresponding embeddings
      _neg_embeddings = (_neg_embeddings * _neg_weight) # [neg, D]
      _pos_embeddings = (_pos_embeddings * _pos_weight) # [pos, D]

      # Calculate attention embeddings for negative and positive samples separately
      _neg_atten = self.get_atten(_neg_embeddings)
      _pos_atten = self.get_atten(_pos_embeddings)
      # Concatenate positive and negative features
      _embedding = torch.cat([_neg_atten, _pos_atten], dim=0) # [2D]

      batch_embedding.append(_embedding)

    batch_embedding = torch.stack(batch_embedding) # [B, 2D]
    # Project to final feature space
    projection = self.proj(batch_embedding) # [B, D]
    return projection

  def get_atten(self, ND):
    # Add classification token to the beginning of patch sequence
    cls_token = self.cls_token.expand(1, -1)
    cls_embedding = torch.cat([cls_token, ND], dim=0).unsqueeze(0) # [1, 1+N, D]
    _z = self.ln(cls_embedding)
    _atten = _z
    # Process features through multiple attention layers
    for atten_layer in self.atten_topk:
      _atten, _ = atten_layer(_atten, _atten, _atten, need_weights=False)
      _atten = self.ln(_z + _atten)
    _atten = self.dropout(_atten)
    
    # Use CLS token as final embedding
    projection = _atten.squeeze(0)[0] # use CLS token as embedding
    return projection