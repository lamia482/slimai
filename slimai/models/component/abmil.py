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
    # Attention mechanism - V branch (tanh activation)
    self.attention_V = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim), 
        torch.nn.Tanh(), 
      )

    if attention == "gated":
      # Gated attention - U branch (sigmoid activation)
      self.attention_U = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim), 
        torch.nn.Sigmoid(), 
      )
    else:
      self.attention_U = None

    # Final attention weight computation
    self.attention_W = torch.nn.Linear(hidden_dim, 1)
    
    return
  
  def forward(self, x): # x: [B, N_hat, K]
    """
    Forward pass of the ABMIL module.
    
    Args:
      x: Input tensor of shape [B, N_hat, K] where:
         B: batch size
         N_hat: number of instances (patches)
         K: feature dimension
         
    Returns:
      batch_embeddings: Aggregated embeddings of shape [B, K]
    """
    batch_embeddings = []
    batch_attentions_weights = []

    for _x in x:
      _x = _x.view(-1, self.input_dim) # convert [N*K] or [N, K] to [N, K]
      N = len(_x)
      embeddings = _x.unsqueeze(0) # [1, N, K]
      
      # Compute attention scores
      A_V = self.attention_V(embeddings) # [1, N, H]
      if self.attention_U is not None:
        # Gated attention mechanism
        A_U = self.attention_U(embeddings) # [1, N, H]
        A = A_V * A_U # [1, N, H] - element-wise multiplication
      else:
        A = A_V # [1, N, H]
        
      # Compute final attention weights
      A_W = self.attention_W(A) # [1, N, 1]
      A_W = self.dropout(A_W)
      A_W = A_W.transpose(1, 2)
      if self.attention_temperature is not None:
        A_W = A_W / self.attention_temperature # scale by temperature
        A_W = A_W.softmax(-1) # [1, 1, N] - normalize with softmax
      else:
        A_W = A_W.sigmoid() # [1, 1, N] - sigmoid activation
        
      batch_attentions_weights.append(A_W.squeeze())
      
      # Compute weighted average of embeddings
      Z = torch.bmm(A_W, embeddings) # [1, 1, K] - batch matrix multiplication
      Z = Z.squeeze() # [K]
      if self.keep_size_hat:
        size_feat = torch.log10(torch.tensor([N]).float()).to(Z.device)  # [1]
        Z = torch.cat([Z, size_feat], dim=0)    # [K+1]
      batch_embeddings.append(Z)

    batch_embeddings = torch.stack(batch_embeddings) # [B, K(+1)]

    return batch_embeddings, batch_attentions_weights
