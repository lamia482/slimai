from typing import Any
from pathlib import Path
import lmdb
from .singleton import singleton_wrapper


def get_cacher(
  cache_dir: str,
  cache_size: int = int(1e12), # 1TB
  engine: str = "lmdb"
):
  return {
    "lmdb": LMDBCache,
  }[engine](
    cache_dir=cache_dir,
    cache_size=cache_size,
  )


@singleton_wrapper
class LMDBCache(object):
  def __init__(self, cache_dir: str, cache_size: int):
    self.cache_dir = Path(cache_dir)
    self.cache_dir.mkdir(parents=True, exist_ok=True)
    self.db = lmdb.open(
      (self.cache_dir / "lmdb").as_posix(), 
      subdir=False,
      map_size=cache_size, 
      readonly=False,
      meminit=False, 
      map_async=True,
    )
    self.cache_size = cache_size
    return

  def __repr__(self) -> str:
    return f"LMDBCache(\n" \
           f"  cache_dir={self.cache_dir}\n" \
           f"  entries={len(self)}\n" \
           f")"

  def __getitem__(self, key: str):
    with self.db.begin(write=False) as txn:
      return txn.get(key.encode())

  def __setitem__(self, key: str, value: Any):
    with self.db.begin(write=True) as txn:
      txn.put(key.encode(), value)

  def __delitem__(self, key: str):
    with self.db.begin(write=True) as txn:
      txn.delete(key.encode())

  def __contains__(self, key: str):
    with self.db.begin(write=False) as txn:
      return txn.get(key.encode()) is not None

  def __len__(self):
    with self.db.begin(write=False) as txn:
      return txn.stat()["entries"]
