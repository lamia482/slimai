import time
from .supervised_dataset import SupervisedDataset
from slimai.helper.help_build import DATASETS
from slimai.helper.help_utils import print_log
from slimai.helper.utils.cache import get_cacher


@DATASETS.register_module()
class MILDataset(SupervisedDataset):
  def __init__(self, *args, 
               use_cache=False, 
               cache_embedding_key="embedding", 
               cache_visual_key="visual",
               **kwargs):
    """
    Args:
      cache_embedding: Whether to cache the embedding of the image.
      cache_embedding_key: The key of the embedding in the data.
    """
    super().__init__(*args, **kwargs)
    self.use_cache = use_cache
    self.cache_embedding_key = cache_embedding_key
    self.cache_visual_key = cache_visual_key
    self.cacher = get_cacher()
    return

  def select_sample(self, item):
    """
    Return Cache as follows:
    {
      "image": None, 
      "meta": {
        "embedding": (B, D),
        "visual": (H, W, 3),
      }, 
      "latency": {
        "embedding_cache_latency": latency,
      }
    }
    """
    file = self.files[item]
    embedding_cache_key = "+".join(map(str, [
      "dataset", self.__class__.__name__, 
      file, str(self.transform), self.cache_embedding_key
    ]))
    visual_cache_key = "+".join(map(str, [
      "dataset", self.__class__.__name__, 
      file, str(self.transform), self.cache_visual_key
    ]))

    if self.use_cache and self.cacher.has(embedding_cache_key) and self.cacher.has(visual_cache_key):
      st = time.time()
      embedding = self.cacher.get(embedding_cache_key)
      latency = time.time() - st
      
      data = dict(
        indice=item,
        image=None, 
        latency=dict(
          embedding_cache_latency=latency,
        ), 
        meta=dict(
          wsi_shrink=None, 
          patch_num=-1, 
          use_cache=self.use_cache,
          visual_file=visual_cache_key,
          embedding=embedding, 
          embedding_key=embedding_cache_key,
          visual_key=visual_cache_key,
        )
      )
      self.load_extra_keys(data, item)
    
    else:
      data = super().select_sample(item)
      data["meta"].update(dict(
        use_cache=self.use_cache,
        visual_file=visual_cache_key,
        embedding=None, 
        embedding_key=embedding_cache_key,
        visual_key=visual_cache_key,
      ))

    return data
