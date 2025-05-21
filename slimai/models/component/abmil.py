import torch
from slimai.helper.help_build import MODELS


__all__ = [
  "ABMIL", 
]

@MODELS.register_module()
class ABMIL(torch.nn.Module):
  def __init__(self, *, input_dim, hidden_dim, dropout=0.1, attention=None, 
               **kwargs):
    super().__init__()
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
  
  def forward(self, x): # x: [B, N_hat, K]
    batch_embeddings = []

    for _x in x:
      _x = _x.view(-1, self.input_dim) # convert [N*K] or [N, K] to [N, K]
      embeddings = _x.unsqueeze(0) # [1, N, K]
      A_V = self.attention_V(embeddings) # [1, N, H]
      if self.attention_U is not None:
        A_U = self.attention_U(embeddings) # [1, N, H]
        A = A_V * A_U # [1, N, H]
      else:
        A = A_V # [1, N, H]
      A_W = self.attention_W(A) # [1, N, 1]
      A_W = self.dropout(A_W)
      A_W = A_W.transpose(1, 2).softmax(-1) # [1, 1, N]
      Z = torch.bmm(A_W, embeddings) # [1, 1, K]
      Z = Z.squeeze() # [K]
      batch_embeddings.append(Z)

    batch_embeddings = torch.stack(batch_embeddings) # [B, K]
    return batch_embeddings
