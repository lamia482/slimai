import hashlib
import time
import mmengine
import torch
from pathlib import Path
from .supervised_dataset import SupervisedDataset
from slimai.helper.help_build import DATASETS, build_model
from slimai.helper.distributed import Distributed
from slimai.helper.common import CACHE_ROOT_DIR


@DATASETS.register_module("MILDataset")
class MILDataset(SupervisedDataset):
  def __init__(self, *args, 
               cache_embedding=False, 
               feature_extractor=None, 
               **kwargs):
    super().__init__(*args, **kwargs)
    self.cache_embedding = cache_embedding
    self.dist = Distributed.create()
    if cache_embedding:
      assert (
        feature_extractor is not None
      ), "Feature extractor is required when cache_embedding is True"
      feature_extractor = build_model(feature_extractor).eval()
      self.feature_extractor = self.dist.prepare_for_distributed(feature_extractor)
    return

  def select_sample(self, item):
    file = self.files[item]
    cache_file = Path(CACHE_ROOT_DIR, "dataset", self.__class__.__name__, "{}-{}.pkl".format(
      hashlib.md5(file.encode(encoding="UTF-8")).hexdigest(), 
      hashlib.md5(str(self.transform).encode(encoding="UTF-8")).hexdigest(), 
    ))
    if self.cache_embedding and cache_file.exists():
      st = time.time()
      data = mmengine.load(cache_file)
      latency = time.time() - st
      data["latency"] = dict(
        embedding_cache_latency=latency,
      )
      return data
    
    data = super().select_sample(item)

    if not self.cache_embedding:
      return data
    
    images = data["image"] # (N, C, H, W)
    data["image"] = None # remove data but keep the key for custom processing

    group_size = 4
    if group_size <= 0:
      group_size = len(images)
    
    embeddings = []

    for i in range(0, len(images), group_size):
      tmp_data = self.dist.prepare_for_distributed(images[i:i+group_size])
      with torch.inference_mode():
        tmp_embeddings = self.feature_extractor(tmp_data) # type: ignore
      embeddings.append(tmp_embeddings)
    embeddings = torch.cat(embeddings, dim=0).cpu()

    data["meta"].update(dict(
      embeddings=embeddings, 
    ))
    mmengine.dump(data, cache_file)
    return data
