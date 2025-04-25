import torch
from slimai.helper.help_build import MODELS


@MODELS.register_module()
class KNNClassifier(object):
  def __init__(self, k, num_classes, metric="cosine", reverse_max=False):
    """
    Args:
      k: int, k value for knn classifier
      metric: str, metric for knn classifier
      reverse_max: bool, whether to set the max metric value to the min metric value to ignore the relevance in future softmax
    """
    self.k = k
    assert (
      (isinstance(self.k, str) and self.k in ["auto", "full", "all"]) or 
      (isinstance(self.k, int) and self.k > 0)
    ), f"k must be a string in ['auto', 'full', 'all'] or an integer greater than 0, but got {self.k}"
    self.num_classes = num_classes
    self.metric = metric
    assert (
      self.metric in ["cosine", "euclidean"]
    ), f"metric must be a string in ['cosine', 'euclidean'], but got {self.metric}"
    self.reverse_max = reverse_max
    self.X = None
    self.Y = None
    self.k_value = None
    return

  def fit(self, X_train, y_train):
    assert (
      y_train.max() < self.num_classes
    ), f"y_train must be less than num_classes({self.num_classes}), but got {y_train.max()}"

    # Store training data for distance calculations
    self.X = X_train
    self.Y = y_train

    if self.k == "auto":
      self.k_value = min(len(self.X), max(5, int(len(self.X) ** 0.5))) # 5 <= k <= sqrt(len(X)) or len(X)
    elif self.k in ["full", "all"]:
      self.k_value = len(self.X)
    else:
      self.k_value = self.k
    return

  def compute_similarity(self, X1, X2):
    """ compute similarity between X and Y
    Args:
      X1: [A, K] embedding of A samples
      X2: [B, K] embedding of B samples
    Returns:
      sim: [A, B] similarity between X1 and X2
    """
    if self.metric == "cosine":
      value_range = [-1, 1]
      sim = torch.cosine_similarity(X1[:, None, ...], X2[None, :, ...], dim=-1)
    elif self.metric == "euclidean":
      value_range = [0, 1]
      sim = torch.cdist(X1, X2, p=2)
    else:
      raise ValueError(f"Unsupported metric: {self.metric}")
    
    if self.reverse_max:
      sim[sim == value_range[1]] = value_range[0]

    return sim

  def predict(self, X, chunk_size=256):
    """
    Args:
      X: [A, K] embedding of A samples
      chunk_size: int, chunk size for similarity computation
    Returns:
      scores: [A, B] softmax scores between X and training samples
    """
    k = self.k_value
    X1, X2, Y = X, self.X, self.Y
    C = self.num_classes

    logits = []
    for i_index in torch.arange(0, len(X1), step=chunk_size):
      col_logits = []
      for j_index in torch.arange(0, len(X2), step=chunk_size):
        _X1 = X1[i_index:i_index+chunk_size, ...]
        _X2 = X2[j_index:j_index+chunk_size, ...]
        sim = self.compute_similarity(_X1, _X2) # [_X1, _X2]
        col_logits.append(sim)
      # N * [chunk_size, chunk_size] -> [chunk_size, X2]
      col_logits = torch.cat(col_logits, dim=-1) # [chunk_size, X2]
      logits.append(col_logits) 
    # N * [chunk_size, X2] -> [X1, X2]
    logits = torch.cat(logits, dim=0) # [X1, X2]
    
    # apply topk weights to Y to obtain scores
    logits_topk, indices_topk = torch.topk(logits, k=k, dim=-1) # [X1, k]
    weights_topk = logits_topk.softmax(dim=-1) # [X1, k]
    Y_topk = Y[indices_topk] # [X1, k]

    # For each sample in X1, distribute the weights to corresponding class indices
    scores = weights_topk.new_zeros(X1.shape[0], C) # [X1, C]
    scores.scatter_add_(dim=-1, index=Y_topk, src=weights_topk) # [X1, C]
    return scores
