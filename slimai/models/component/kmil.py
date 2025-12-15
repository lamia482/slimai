import torch
from slimai.helper.help_build import MODELS


__all__ = [
  "KMIL", 
]

@MODELS.register_module()
class KMIL(torch.nn.Module):
  """
  Knowledge-aware Multiple Instance Learning
  compute attention(weight) for each patch, then recursively 
  re-compute attention for the topk-percent patches applied with attention weight 
  untill the final one embedding vector. 
  The final embedding vector is used as the representation of the instance.
  """

  MLP: torch.nn.Module = MODELS.get("MLP") # type: ignore

  def __init__(self, *, input_dim, num_heads, num_layers, dropout=0.1, topk_percent=0.3, 
               act="gelu", norm="batch_norm_1d", **kwargs):
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
    self.dropout = torch.nn.Dropout(dropout)
    self.topk_percent = topk_percent  # Percentage of top-k patches to keep in each iteration
    return
  
  def forward(self, x):
    batch_embeddings = []
    batch_attentions_weights = []

    # Process each sample in the batch individually
    for _x in x:
      _embedding, _weights = self.compute(_x)
      batch_embeddings.append(_embedding)
      batch_attentions_weights.append(_weights)

    batch_embeddings = torch.stack(batch_embeddings) # [B, D]
    projection = self.proj(batch_embeddings) # [B, D]
    return projection, batch_attentions_weights

  def compute(self, x):
    """ TODO: divide num_classes clusters, then compute attention for each cluster
    """
    weights = self.atten_weight(x) # [N, D] -> [N, 1]
    weights = weights.squeeze().sigmoid() # [N]

    # determine topk
    if self.topk_percent >= 1 and isinstance(self.topk_percent, int):
      topk = self.topk_percent
    else:
      topk = max(1, int(weights.shape[0] * self.topk_percent))
    
    topk_scores, topk_indices = weights.topk(k=topk, dim=-1, largest=True) # select topk(maximum) indices

    x = x[topk_indices] * topk_scores.unsqueeze(-1) # [topk, D] # type: ignore
    return x.mean(dim=0), weights