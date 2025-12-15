from typing import Any, Tuple, Optional, Iterator
from pathlib import Path
import json
import os
import hashlib
import shutil
from enum import Enum
import io
import lmdb
import numpy as np
import torch
import zlib
from PIL import Image as PILImageHelper
from PIL.Image import Image as PILImage
from torchvision.tv_tensors import Image as TVImage
import mmengine
from .singleton import singleton_wrapper
from .select import chunks
from . import async_task


__all__ = ["get_cacher"]

CACHE_ROOT_DIR = "/.slimai/cache"

def get_cacher(
  engine: str = "lmdb",
  cache_dir: str = CACHE_ROOT_DIR,
  cache_size: int = -1, # use all storage space
):
  return {
    "lmdb": LMDBCache,
  }[engine](
    cache_dir=cache_dir,
    cache_size=cache_size,
  )


class DataType(Enum):
  String = str(str)
  Tensor = str(torch.Tensor)
  Numpy = str(np.ndarray)
  TVImage = str(TVImage)
  PILImage = str(PILImage)
  Json = str(dict)


class CacheItem(object):
  def __init__(self, dtype: Optional[str], data: Optional[bytes], inp_key: str, hash_key: str):
    self.dtype = dtype
    self.data = data
    self.inp_key = inp_key
    self.hash_key = hash_key
    return

  def __repr__(self) -> str:
    return f"CacheItem(\n" \
           f"  dtype={self.dtype}\n" \
           f"  inp_key={self.inp_key}\n" \
           f"  hash_key={self.hash_key}\n" \
           f")"


def get_size(start_path = '.'):
  total_size = 0
  for dirpath, dirnames, filenames in os.walk(start_path):
    for f in filenames:
      fp = os.path.join(dirpath, f)
      # skip if it is symbolic link
      if not os.path.islink(fp):
          total_size += os.path.getsize(fp)
  return total_size


@singleton_wrapper
class LMDBCache(object):
  def __init__(self, 
               cache_dir: str, cache_size: int, 
               byte_split_limit_size: int=256, 
               compress_level: int=0
    ):
    self.cache_dir = Path(cache_dir) / "lmdb"
    self.lmdb_data_name = "database"
    self.lmdb_meta_name = "meta"
    self.mess_dir = self.cache_dir / "mess"
    os.makedirs(self.mess_dir, exist_ok=True)

    self.byte_split_limit_size = byte_split_limit_size

    if cache_size <= 0:
      cache_size = int(0.9 * (shutil.disk_usage(self.cache_dir).free + get_size(self.cache_dir.as_posix())))

    self.env_kwargs = dict(
      map_size=cache_size, 
      subdir=False,
      readonly=False,
      metasync=True,
      sync=True, 
      writemap=False,
      map_async=True, 
      max_dbs=3,
    )

    self.db_kwargs = dict(
      dupsort=True,
      dupfixed=False,
    )

    self.data_env, self.database = self.create_env(self.lmdb_data_name)
    self.meta_env, self.metabase = self.create_env(self.lmdb_meta_name)

    self.compress_level = compress_level
    return

  def __repr__(self) -> str:
    return f"LMDBCache(\n" \
           f"  cache_dir={self.cache_dir}\n" \
           f"  data_length={self.data_length}\n" \
           f"  meta_length={self.meta_length}\n" \
           f")"

  def get(self, key: str, no_wait: bool = False):
    if no_wait:
      return async_task.submit_async_task(self.kernel_get, key)
    return self.kernel_get(key)

  def put(self, key: str, value: Any, no_wait: bool = False):
    if no_wait:
      return async_task.submit_async_task(self.kernel_put, key, value)
    return self.kernel_put(key, value)

  def has(self, key: str):
    return (key in self)

  def kernel_get(self, key: str):
    item = self[key]
    cache_file = self.mess_dir / f"{item.hash_key}.pkl"
    
    if item.data is None and cache_file.exists():
      return mmengine.load(cache_file)

    if item.dtype is None:
      return None
    
    value = self.decode_value(item.dtype, item.data) # type: ignore

    return value

  def kernel_put(self, key: str, value: Any):
    dtype = str(type(value))
    
    try:
      vbytes = self.encode_value(dtype, value) # try serialize
      self[key] = (dtype, vbytes)
    except NotImplementedError:
      hash_key = self.translate_key(key)
      mmengine.dump(value, self.mess_dir / f"{hash_key}.pkl", protocol=4) # try pickle
    except Exception as ex:
      raise ex
    return

  def __getitem__(self, key: str) -> CacheItem:
    hash_key = self.translate_key(key)

    if not self.has(key):
      return CacheItem(None, None, key, hash_key) # return None

    kbytes = hash_key.encode()

    def _get_bytes_(env, db, k) -> bytes:
      with env.begin(db=db, write=False) as txn, txn.cursor(db=db) as cursor:
        cursor.set_key(k)
        index_list, value_list = zip(*[
          iv.split(b"_", maxsplit=1) for _, iv in enumerate(cursor.iternext_dup()) # type: ignore
        ])

        value_list = list(value_list)

        sorted_indices = np.argsort(list(map(int, index_list)))
        sorted_vbytes_list = [value_list[i] for i in sorted_indices]
        return b"".join(sorted_vbytes_list)

    data = _get_bytes_(self.data_env, self.database, kbytes)
    dtype, inp_key = _get_bytes_(self.meta_env, self.metabase, kbytes).decode().split(":", maxsplit=1)
    assert (
      inp_key == key
    ), f"Input key mismatch: {inp_key} != {key}"

    return CacheItem(dtype, data, key, hash_key)

  def __setitem__(self, key: str, data: Tuple[str, bytes]):
    if self.has(key):
      del self[key]

    hash_key = self.translate_key(key)
    kbytes = hash_key.encode()
    dtype, value = data

    def _set_bytes_(env, db, k, seq) -> None:
      seq = list(seq)
      nbit = np.ceil(np.log10(len(seq))).astype("int")
      with env.begin(db=db, write=True) as txn, txn.cursor(db=db) as cursor:
        for i, v in enumerate(seq):
          cursor.put(k, f"{i:0{nbit}d}_".encode() + v, dupdata=True) # add index and length to value for futher sort
      env.close()
      return

    _set_bytes_(*self.create_env(self.lmdb_data_name), 
                kbytes, self.split_bytes_into_seq(value))
    _set_bytes_(*self.create_env(self.lmdb_meta_name), 
                kbytes, self.split_bytes_into_seq(f"{dtype}:{key}".encode()))
    return

  def __delitem__(self, key: str):
    hash_key = self.translate_key(key)
    kbytes = hash_key.encode()
    with self.data_env.begin(db=self.database, write=True) as txn:
      txn.delete(kbytes)
    with self.meta_env.begin(db=self.metabase, write=True) as txn:
      txn.delete(kbytes)
    return

  def __contains__(self, key: str):
    hash_key = self.translate_key(key)
    kbytes = hash_key.encode()
    with self.data_env.begin(db=self.database, write=False) as txn:
      has_database = txn.get(kbytes) is not None
    with self.meta_env.begin(db=self.metabase, write=False) as txn:
      has_meta = txn.get(kbytes) is not None
    return has_database and has_meta

  def create_env(self, db_name: str, env_name: str = "data"):
    env = lmdb.open((self.cache_dir / env_name).as_posix(), **self.env_kwargs)
    db = env.open_db(db_name.encode(), **self.db_kwargs)
    return env, db

  @property
  def data_length(self):
    return len(self.data_keys)
  
  @property
  def meta_length(self):
    return len(self.meta_keys)

  @property
  def data_keys(self):
    with self.data_env.begin(db=self.database, write=False) as txn, txn.cursor(db=self.database) as cursor:
      return [key.decode() for key in cursor.iternext_nodup(keys=True, values=False)]

  @property
  def meta_keys(self):
    with self.meta_env.begin(db=self.metabase, write=False) as txn, txn.cursor(db=self.metabase) as cursor:
      return [key.decode() for key in cursor.iternext_nodup(keys=True, values=False)]

  def split_bytes_into_seq(self, value: bytes) -> Iterator[bytes]:
    return chunks(value, self.byte_split_limit_size)

  def translate_key(self, key: str):
    assert (
      isinstance(key, str)
    ), f"Key must be a string, but got: {type(key)}"
    key = hashlib.sha256(key.encode()).hexdigest()
    return key

  def encode_value(self, dtype: str, value: Any) -> bytes:
    def _encode(_value: Any) -> bytes:
      match dtype:
        case DataType.Tensor.value:
          buffer = io.BytesIO()
          torch.save(_value.detach().cpu(), buffer)
          return buffer.getvalue()
        case DataType.Numpy.value:
          buffer = io.BytesIO()
          np.save(buffer, _value)
          return buffer.getvalue()
        case DataType.TVImage.value:
          buffer = io.BytesIO()
          torch.save(_value.detach().cpu(), buffer)
          return buffer.getvalue()
        case DataType.PILImage.value:
          buffer = io.BytesIO()
          _value.save(buffer, format="PNG")
          return buffer.getvalue()
        case DataType.String.value:
          buffer = io.BytesIO()
          np.save(buffer, _value.encode())
          return buffer.getvalue()
        case DataType.Json.value:
          buffer = io.BytesIO()
          np.save(buffer, json.dumps(_value, ensure_ascii=False).encode())
          return buffer.getvalue()
        case _:
          raise NotImplementedError(f"Unsupported dtype: {dtype}")
          
    vbytes = _encode(value)
    if self.compress_level != 0:
      return zlib.compress(vbytes, self.compress_level)
    return vbytes

  def decode_value(self, dtype: str, value: bytes) -> Any:
    def _decode(_vbytes: bytes) -> Any:
      match dtype:
        case DataType.Tensor.value:
          buffer = io.BytesIO(_vbytes)
          return torch.load(buffer, weights_only=False)
        case DataType.Numpy.value:
          buffer = io.BytesIO(_vbytes)
          return np.load(buffer)
        case DataType.TVImage.value:
          buffer = io.BytesIO(_vbytes)
          return TVImage(torch.load(buffer, weights_only=False))
        case DataType.PILImage.value:
          return PILImageHelper.open(io.BytesIO(_vbytes))
        case DataType.String.value:
          buffer = io.BytesIO(_vbytes)
          return np.load(buffer).item().decode()
        case DataType.Json.value:
          buffer = io.BytesIO(_vbytes)
          return json.loads(np.load(buffer).item())
        case _:
          raise NotImplementedError(f"Unsupported dtype: {dtype}")
        
    if self.compress_level != 0:
      data = zlib.decompress(value)
    else:
      data = value
    return _decode(data)
