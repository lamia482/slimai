from typing import Tuple

import torch
from slimai.helper.help_build import MODELS


__all__ = [
  "ABMIL",
]

@MODELS.register_module()
class ABMIL(torch.nn.Module):
  """
  Attention-Based Multiple Instance Learning (ABMIL) module.
  
  This module implements attention-based pooling for multiple instance learning,
  which assigns attention weights to instances and computes a weighted average.
  
  References:
    - Ilse, M., Tomczak, J., & Welling, M. (2018). Attention-based deep multiple instance learning.
      In International conference on machine learning (pp. 2127-2136).
  """
  def __init__(self, *, input_dim, hidden_dim,
               dropout=0.1, attention=None,
               keep_size_hat=False, attention_temperature=1.0,
               **kwargs):
    """
    Initialize the ABMIL module.
    
    Args:
      input_dim: Dimension of input features
      hidden_dim: Dimension of hidden layer in attention mechanism
      dropout: Dropout rate for regularization
      attention: Type of attention mechanism ('gated' or None)
      **kwargs: Additional arguments
    """
    super().__init__()
    self.keep_size_hat = keep_size_hat
    self.attention_temperature = attention_temperature
    self.input_dim = input_dim
    self.dropout = torch.nn.Dropout(dropout)
    assert (
      attention in [None, "gated"]
    ), "attention must be None or 'gated', but got {}".format(attention)
    self.attention = attention
    self.attention_V = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.Tanh(),
      )

    if attention == "gated":
      self.attention_U = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.Sigmoid(),
      )
    else:
      self.attention_U = None

    self.attention_W = torch.nn.Linear(hidden_dim, 1)

    return

  def _forward_single_bag(self, _x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _x = _x.view(-1, self.input_dim)
    num_patches = _x.shape[0]
    embeddings = _x.unsqueeze(0)
    A_V = self.attention_V(embeddings)
    if self.attention_U is not None:
      A_U = self.attention_U(embeddings)
      A = A_V * A_U
    else:
      A = A_V
    A_W = self.attention_W(A)
    A_W = self.dropout(A_W)
    A_W = A_W.transpose(1, 2)
    if self.attention_temperature is not None:
      A_W = A_W / self.attention_temperature
      A_W = A_W.softmax(-1)
    else:
      A_W = A_W.sigmoid()
    attention_weights = A_W.squeeze(0).squeeze(0)
    bag_embedding = torch.bmm(A_W, embeddings).squeeze(0).squeeze(0)
    if self.keep_size_hat:
      size_feat = torch.log10(torch.tensor([num_patches], device=bag_embedding.device).float())
      bag_embedding = torch.cat([bag_embedding, size_feat], dim=0)
    return bag_embedding, attention_weights

  def forward(self, x):
    """
    Forward pass of the ABMIL module.
    
    Args:
      x: list of bags, each [N, K] or flattenable to [N, K]
         
    Returns:
      batch_embeddings: [B, K(+1)]
      batch_attentions_weights: list of attention tensors
    """
    batch_embeddings = []
    batch_attentions_weights = []
    for _x in x:
      bag_embedding, attention_weights = self._forward_single_bag(_x)
      batch_embeddings.append(bag_embedding)
      batch_attentions_weights.append(attention_weights)
    return torch.stack(batch_embeddings), batch_attentions_weights

  def export_model(self) -> torch.nn.Module:
    return _ABMILExport(self).eval()


class _ABMILExport(torch.nn.Module):
  """Single-bag ABMIL export: input ``embedding_arr [N,K]``."""

  def __init__(self, source: ABMIL):
    super().__init__()
    self.abmil = source
    return

  def forward(self, embedding_arr: torch.Tensor):
    return self.abmil._forward_single_bag(embedding_arr)
