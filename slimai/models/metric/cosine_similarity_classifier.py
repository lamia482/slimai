import torch
from slimai.helper.help_build import MODELS


@MODELS.register_module()
class CosineSimilarityClassifier:
  def __init__(self, k):
    self.k = k
    self.pairs = None
    return

  def fit(self, X_train, y_train):
    # Store training data for distance calculations
    assert (
      y_train.max() < self.k
    ), f"y_train.max() = {y_train.max()} must be less than k = {self.k}"
    self.pairs = [
      X_train[y_train == cid]
      for cid in range(self.k)
    ]
    return

  def predict(self, X):
    scores = []
    for pair in self.pairs:
      sim = torch.cosine_similarity(pair[None, ...], X[:, None, ...], dim=-1).mean(dim=-1)
      scores.append(sim)
    logits = torch.stack(scores, dim=-1)
    scores = logits.softmax(dim=-1)
    return scores
