import hashlib
import time
import mmengine
from pathlib import Path
from .supervised_dataset import SupervisedDataset
from slimai.helper.help_build import DATASETS
from slimai.helper.common import CACHE_ROOT_DIR


@DATASETS.register_module("MILDataset")
class MILDataset(SupervisedDataset):
  def __init__(self, *args, 
               cache_embedding=False, 
               cache_embedding_key="embeddings", 
               **kwargs):
    """
    Args:
      cache_embedding: Whether to cache the embedding of the image.
      cache_embedding_key: The key of the embedding in the data.
    """
    super().__init__(*args, **kwargs)
    self.cache_embedding = cache_embedding
    self.cache_embedding_key = cache_embedding_key
    return

  def select_sample(self, item):
    """
    Return Cache as follows:
    {
      "image": None, 
      "meta": {
        "embeddings": [
          (B, D),
        ]
      }, 
      "latency": {
        "embedding_cache_latency": latency,
      }
    }
    """
    file = self.files[item]
    cache_file = Path(CACHE_ROOT_DIR, "dataset", self.__class__.__name__, "{}-{}.pkl".format(
      hashlib.md5(file.encode(encoding="UTF-8")).hexdigest(), 
      hashlib.md5(str(self.transform).encode(encoding="UTF-8")).hexdigest(), 
    ))
    vis_cache_file = Path(cache_file.as_posix().replace(".pkl", "_vis.pkl"))

    if self.cache_embedding and cache_file.exists() and vis_cache_file.exists():
      st = time.time()
      data = mmengine.load(cache_file)
      latency = time.time() - st
      
      assert (
        self.cache_embedding_key in data["meta"]
      ), f"Cache embedding key {self.cache_embedding_key} not found in data['meta']"
      
      data.update(dict(
        indice=item,
        image=None, 
        latency=dict(
          embedding_cache_latency=latency,
        ), 
      ))
      data["meta"].update(dict(
        cache_embedding=self.cache_embedding,
        visual_file=vis_cache_file.as_posix(),
        cache_file=cache_file.as_posix(),
      ))
      self.load_extra_keys(data, item)
    
    else:
      data = super().select_sample(item)
      data["meta"].update(dict(
        cache_embedding=self.cache_embedding,
        visual_file=vis_cache_file.as_posix(),
        cache_file=cache_file.as_posix(),
      ))

    return data
