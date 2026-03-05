import numpy as np
import torch

from slimai.helper.help_build import TRANSFORMS
from .base_transform import BaseTransform


@TRANSFORMS.register_module()
class RandomSelectFeatureBag(object):
  def __init__(self, low=0.3, high=0.9, p=1.0, min_keep=16):
    self.low = low
    self.high = high
    self.p = p
    self.min_keep = min_keep
    return

  def __call__(self, embeddings: torch.Tensor) -> torch.Tensor:
    if torch.rand([]) > self.p:
      return embeddings
    ratio = np.random.uniform(self.low, self.high)
    if ratio < 1:
      count = int(len(embeddings) * ratio)
    else:
      count = int(ratio)
    count = max(count, self.min_keep)
    count = min(count, len(embeddings))
    return embeddings[:count]


@TRANSFORMS.register_module()
class RandomShuffleFeatureBag(object):
  def __init__(self, p=0.5):
    self.p = p
    return

  def __call__(self, embeddings: torch.Tensor) -> torch.Tensor:
    if torch.rand([]) > self.p:
      return embeddings
    indices = torch.randperm(len(embeddings), device=embeddings.device)
    return embeddings[indices]


@TRANSFORMS.register_module()
class RandomFeatureDropout(object):
  def __init__(self, p=0.1, q=0.1):
    self.p = p
    self.q = q
    return

  def __call__(self, embeddings: torch.Tensor) -> torch.Tensor:
    if torch.rand([]) > self.p:
      return embeddings
    mask = (torch.rand_like(embeddings) > self.q).float()
    return embeddings * mask / max(1e-6, (1 - self.q))


@TRANSFORMS.register_module()
class AddGaussianNoise(object):
  def __init__(self, std=None, p=0.1):
    self.std = std
    self.p = p
    return

  def __call__(self, embeddings: torch.Tensor) -> torch.Tensor:
    if torch.rand([]) > self.p:
      return embeddings
    std = torch.std(embeddings) if self.std is None else self.std
    noise = torch.rand_like(embeddings) * std
    mask = torch.sign(torch.rand_like(embeddings) - 0.5)
    if self.std is None:
      return embeddings * (1 - mask) + noise * mask
    return embeddings + noise * mask


@TRANSFORMS.register_module()
class MixupPatches(object):
  def __init__(self, alpha=0.1, p=0.1):
    self.alpha = alpha
    self.p = p
    return

  def __call__(self, embeddings: torch.Tensor) -> torch.Tensor:
    if torch.rand([]) > self.p:
      return embeddings
    if embeddings.size(0) < 2:
      return embeddings
    lam = torch.from_numpy(
      np.random.beta(self.alpha, self.alpha, size=[len(embeddings), 1])
    ).float().to(embeddings.device)
    indices = torch.randperm(len(embeddings), device=embeddings.device)
    return lam * embeddings + (1 - lam) * embeddings[indices]


@TRANSFORMS.register_module()
class EmbeddingAugmenter(BaseTransform):
  def __init__(self, transforms=None, embedding_key="embedding"):
    if transforms is None:
      transforms = [
        dict(type="RandomSelectFeatureBag", low=0.3, high=0.9, p=1.0),
        dict(type="RandomShuffleFeatureBag", p=0.5),
        dict(type="RandomFeatureDropout", p=0.1, q=0.1),
        dict(type="AddGaussianNoise", std=None, p=0.1),
        dict(type="MixupPatches", alpha=0.1, p=0.1),
      ]
    self.embedding_key = embedding_key
    super().__init__(transforms=transforms)
    return

  def __call__(self, data):
    meta = data.get("meta", None)
    if meta is None:
      return data
    embedding = meta.get(self.embedding_key, None)
    if embedding is None:
      return data
    meta[self.embedding_key] = self.transforms(embedding)
    data["meta"] = meta
    return data

  def compose(self, transforms):
    transforms = self._compose(
      transforms=transforms,
      source=[TRANSFORMS],
      recursive_key="transforms",
    )
    if not isinstance(transforms, (tuple, list)):
      transforms = [transforms]

    def wrap(embedding):
      output = embedding
      for transform in transforms:
        output = transform(output)
      return output

    return wrap

