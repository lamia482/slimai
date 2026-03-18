import torch
from slimai.helper.help_build import MODELS


__all__ = [
  "KMIL", "WMIL", "SortWMIL", 
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

@MODELS.register_module()
class WMIL(torch.nn.Module):
  def __init__(self, *, input_dim, hidden_dim, 
               dropout=0.1, attention=None, 
               keep_size_hat=False, attention_temperature=1.0,
               **kwargs):
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
      else:
        A_W = A_W.sigmoid() # [1, 1, N] - sigmoid activation

      k = 10
      _, topk_indices = A_W.topk(k=k, dim=-1, largest=True) # select topk(maximum) indices
      _, tailk_indices = A_W.topk(k=k, dim=-1, largest=False) # select tailk(minimum) indices
      indices = torch.cat([topk_indices, tailk_indices], dim=-1).squeeze()

      truncate_A_W = A_W[:, :, indices]
      truncate_E = embeddings[:, indices, :]

      truncate_A_W = truncate_A_W.softmax(-1)

      attens = [None] * N

      for ik, iv in zip(indices.squeeze(), truncate_A_W.squeeze()):
        attens[ik] = iv
        
      batch_attentions_weights.append(attens)
      
      # Compute weighted average of embeddings
      Z = torch.bmm(truncate_A_W, truncate_E) # [1, 1, K] - batch matrix multiplication
      Z = Z.squeeze() # [K]
      if self.keep_size_hat:
        size_feat = torch.log10(torch.tensor([N]).float()).to(Z.device)  # [1]
        Z = torch.cat([Z, size_feat], dim=0)    # [K+1]
      batch_embeddings.append(Z)

    batch_embeddings = torch.stack(batch_embeddings) # [B, K(+1)]

    return batch_embeddings, batch_attentions_weights


@MODELS.register_module()
class SortWMIL(WMIL):
  def __init__(self, *, input_dim, hidden_dim, 
               dropout=0.1, attention=None, 
               keep_size_hat=False, attention_temperature=1.0,
               **kwargs):
    super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, 
                     dropout=dropout, attention=attention, 
                     keep_size_hat=keep_size_hat, attention_temperature=attention_temperature,
                     **kwargs)
    return
  
  def forward(self, x): # x: [B, N_hat, K]
    batch_embeddings = []
    batch_attentions_weights = []

    batch_topk_embeddings, batch_tailk_embeddings = [], []
    batch_topk_logits, batch_tailk_logits = [], []

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
      else:
        A_W = A_W.sigmoid() # [1, 1, N] - sigmoid activation

      k = 10
      _, topk_indices = A_W.topk(k=k, dim=-1, largest=True) # select topk(maximum) indices
      _, tailk_indices = A_W.topk(k=k, dim=-1, largest=False) # select tailk(minimum) indices
      indices = torch.cat([topk_indices, tailk_indices], dim=-1).squeeze()

      truncate_A_W = A_W[:, :, indices]
      truncate_E = embeddings[:, indices, :]

      batch_topk_embeddings.append(truncate_E[0, :k]) # append [k, K]
      batch_tailk_embeddings.append(truncate_E[0, k:]) # append [k, K]
      batch_topk_logits.append(truncate_A_W[0, 0, :k]) # append [k]
      batch_tailk_logits.append(truncate_A_W[0, 0, k:]) # append [k]

      truncate_A_W = truncate_A_W.softmax(-1)

      attens = [None] * N

      for ik, iv in zip(indices.squeeze(), truncate_A_W.squeeze()):
        attens[ik] = iv
        
      batch_attentions_weights.append(attens)
      
      # Compute weighted average of embeddings
      Z = torch.bmm(truncate_A_W, truncate_E) # [1, 1, K] - batch matrix multiplication
      Z = Z.squeeze() # [K]
      if self.keep_size_hat:
        size_feat = torch.log10(torch.tensor([N]).float()).to(Z.device)  # [1]
        Z = torch.cat([Z, size_feat], dim=0)    # [K+1]
      batch_embeddings.append(Z)

    batch_embeddings = torch.stack(batch_embeddings) # [B, K(+1)]
    batch_topk_embeddings = torch.stack(batch_topk_embeddings) # [B, k, K]
    batch_tailk_embeddings = torch.stack(batch_tailk_embeddings) # [B, k, K]
    batch_topk_logits = torch.stack(batch_topk_logits)
    batch_tailk_logits = torch.stack(batch_tailk_logits)

    return batch_embeddings, batch_attentions_weights, dict(
      topk_embeddings=batch_topk_embeddings, tailk_embeddings=batch_tailk_embeddings,
      topk_logits=batch_topk_logits, tailk_logits=batch_tailk_logits
    )
  

@MODELS.register_module()
class THCAHeadC3(torch.nn.Module):
  """ THCAHeadC3: A head for THCA classification with 3 classes (C0, C1, C2)
  The head consists of two separate MLPs:
  - cls_head: for classifying into 3 classes (C0, C1, C2)
  - C1_offset_head: for predicting the offset of C1 (the probability of being C1)
  The final logits are computed by combining the outputs of both heads, where the probability of C1 is adjusted by the offset predicted by C1_offset_head.
  """
  MLP: torch.nn.Module = MODELS.get("MLP") # type: ignore

  def __init__(self, *, 
               input_dim, output_dim, hidden_dim=2048, bottleneck_dim=512, 
               n_layer=3, act="gelu", norm=None, dropout=None):
    assert (
      output_dim == 3
    ), "output_dim must be 3 for THCAHeadC3, but got {}".format(output_dim)
    super().__init__()
    self.cls_head = self.MLP(input_dim=input_dim, output_dim=3, 
                             hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim, 
                             n_layer=n_layer, act=act, norm=norm, dropout=dropout)
    self.C1_offset_head = self.MLP(input_dim=input_dim, output_dim=1, 
                                   hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim, 
                                   n_layer=n_layer, act=act, norm=norm, dropout=dropout)
    return
  
  def forward(self, x):
    c1_logits = self.C1_offset_head(x) # [B, 1]
    cls_logits = self.cls_head(x) # [B, 3]

    cls_prob = cls_logits.softmax(dim=-1) # [B, 3]
    c1_prob = c1_logits.sigmoid() # [B, 1]

    final_prob = cls_prob.clone() # [B, 3]
    final_prob[:, 1] += c1_prob.squeeze()
    final_logits = torch.log(final_prob + 1e-8)

    return final_logits


@MODELS.register_module()
class THCAHeadC3BRAF(THCAHeadC3):
  """ THCAHeadC3: A head for THCA classification with 3 classes (C0, C1, C2)
  The head consists of two separate MLPs:
  - cls_head: for classifying into 3 classes (C0, C1, C2)
  - C1_offset_head: for predicting the offset of C1 (the probability of being C1)
  The final logits are computed by combining the outputs of both heads, where the probability of C1 is adjusted by the offset predicted by C1_offset_head.
  """
  MLP: torch.nn.Module = MODELS.get("MLP") # type: ignore

  def __init__(self, *, 
               input_dim, output_dim, hidden_dim=2048, bottleneck_dim=512, 
               n_layer=3, act="gelu", norm=None, dropout=None):
    super().__init__(input_dim=input_dim, output_dim=output_dim, 
                     hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim, 
                     n_layer=n_layer, act=act, norm=norm, dropout=dropout)
    self.braf_head = self.MLP(input_dim=input_dim, output_dim=2, 
                              hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim, 
                              n_layer=n_layer, act=act, norm=norm, dropout=dropout)
    return
  
  def forward(self, x):
    braf_logits = self.braf_head(x) # [B, 2]
    c3_logits = super().forward(x) # [B, 3]
    return dict(
      c3_logits=c3_logits,
      braf_logits=braf_logits,
    )

@MODELS.register_module()
class THCAHeadC3BRAFAware(THCAHeadC3BRAF):
  def forward(self, x):
    """
    基于条件概率融合BRAF预测与C3三分类预测：
      - C3分类: C0, C1, C2
      - BRAF分类: BRAF-, BRAF+
      新策略：
        P(C2|X) = P(C2|X) * P(BRAF+|X)
        P(C0|X) = P(C0|X) * P(BRAF-|X)
        （P(C1|X)可直接用原始C1概率，或做归一化）
      最终归一化得到调整后的三分类概率。
    """
    rst = super().forward(x)
    braf_logits = rst["braf_logits"]    # [B, 2]
    c3_logits = rst["c3_logits"]        # [B, 3]

    braf_prob = braf_logits.softmax(dim=-1)  # [B, 2]
    c3_prob = c3_logits.softmax(dim=-1)      # [B, 3]

    # 条件概率融合
    new_c3_prob = c3_prob.clone()
    new_c3_prob[:, 0] = c3_prob[:, 0] * braf_prob[:, 0]  # C0 × BRAF-
    new_c3_prob[:, 2] = c3_prob[:, 2] * braf_prob[:, 1]  # C2 × BRAF+
    # 可选地，C1不结合BRAF，仅靠自身置信
    new_c3_prob[:, 1] = c3_prob[:, 1] * braf_prob[:, 0]
    # 最终归一化
    new_c3_prob = new_c3_prob / (new_c3_prob.sum(dim=-1, keepdim=True) + 1e-8)
    new_c3_logits = torch.log(new_c3_prob + 1e-8)

    return dict(
      c3_logits=new_c3_logits,
      braf_logits=braf_logits,
    )


# ✅ 解决方案：动态权重（Dynamic Weighting）
# 参考：Multi-Task Learning Using Uncertainty to Weigh Losses (CVPR 2018)
@MODELS.register_module()
class DynamicWeighting(torch.nn.Module):
  """ DynamicWeighting: A module for dynamically weighting the loss of two tasks
  The loss of two tasks are weighted by the precision of the loss, which is estimated by the variance of the loss.
  The precision is estimated by the variance of the loss, which is estimated by the variance of the loss.
  The precision is estimated by the variance of the loss, which is estimated by the variance of the loss.
  """
  def __init__(self):
    super().__init__()
    self.log_var_A = torch.nn.Parameter(torch.tensor(0.))
    self.log_var_B = torch.nn.Parameter(torch.tensor(0.))
    return
  
  def forward(self, loss_A, loss_B):
    precision_A = torch.exp(-self.log_var_A)
    precision_B = torch.exp(-self.log_var_B)
    
    weighted_loss_A = precision_A * loss_A + 0.5 * self.log_var_A
    weighted_loss_B = precision_B * loss_B + 0.5 * self.log_var_B
    
    return weighted_loss_A + weighted_loss_B
