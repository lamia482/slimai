import os
import hashlib
import mmengine
import numpy as np
from typing import Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sdk.reader import get_reader_by_file
from slimai.helper.help_build import LOADERS
from slimai.helper.help_utils import _CACHE_ROOT_DIR_, dist_env, print_log


@LOADERS.register_module()
class RegionTileLoader():
  def __init__(self, *, 
               magnification: int, 
               region: Dict, 
               cache: bool=True, 
               num_threads: int=None, 
               padding_value: int=255,
               ):
    self.magnification = magnification
    self.region = region
    assert (
      isinstance(region, dict) and {"xmin", "ymin", "xmax", "ymax"}.issubset(set(region.keys()))
    ), "Region must be a dictionary with keys: 'xmin', 'ymin', 'xmax', 'ymax'"
    self.cache = cache
    self.num_threads = num_threads
    self.padding_value = padding_value
    return
  
  def __call__(self, file):
    cache_file = Path(_CACHE_ROOT_DIR_, "loader", self.__class__.__name__, "{}-{}.pkl".format(
      hashlib.md5("+".join(map(str, [self.magnification, self.region])).encode(encoding="UTF-8")
    ).hexdigest(), file))
    if self.cache and cache_file.exists():
      return mmengine.load(cache_file)
    
    wsi_file_path = file

    reader = get_reader_by_file(wsi_file_path, scale=self.magnification)
    if not reader.status:
      return None
    
    xmin, ymin, xmax, ymax = [self.region[k] for k in ["xmin", "ymin", "xmax", "ymax"]]
    if xmax == -1:
      xmax = reader.getReadWidth()
    if ymax == -1:
      ymax = reader.getReadHeight()
    
    if self.num_threads:
      tile = self.read_roi_async(reader, xmin, ymin, xmax, ymax, self.num_threads)
    else:
      tile = reader.ReadRoi(xmin, ymin, xmax-xmin, ymax-ymin, scale=reader.getReadScale())

    if self.cache:
      mmengine.dump(tile, cache_file)
    
    return tile

  def read_roi_async(self, reader, xmin, ymin, xmax, ymax, num_threads):
    image = np.full((ymax-ymin, xmax-xmin, 3), self.padding_value, dtype=np.uint8)
    
    # Calculate number of tiles needed
    x_chunks = np.ceil(np.sqrt(num_threads)).astype(np.int32)
    x_tile_size = np.ceil((xmax - xmin) / x_chunks).astype(np.int32)
    y_chunks = np.ceil(np.sqrt(num_threads)).astype(np.int32)
    y_tile_size = np.ceil((ymax - ymin) / y_chunks).astype(np.int32)
    

    # Create readers upfront for each thread
    thread_readers = []
    for _ in range(num_threads):
      thread_reader = get_reader_by_file(reader.file, scale=reader.getReadScale())
      thread_readers.append(thread_reader)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
      def read_roi_chunk(x, y, w, h, thread_id):
        if w <= 0 or h <= 0:
          return
        # Get reader from thread's reader list
        thread_reader = thread_readers[thread_id]
        try:
          chunk = thread_reader.ReadRoi(x, y, w, h, scale=thread_reader.getReadScale())
        except Exception as e:
          chunk = np.full((h, w, 3), self.padding_value, dtype=np.uint8)
        image[y:y+h, x:x+w, :] = chunk
        return

      futures = []
      # Create grid of chunks
      thread_counter = 0
      for i in range(y_chunks):
        start_y = ymin + i * y_tile_size
        end_y = min(start_y + y_tile_size, ymax)
        
        for j in range(x_chunks):
          start_x = xmin + j * x_tile_size
          end_x = min(start_x + x_tile_size, xmax)
          
          future = executor.submit(read_roi_chunk, start_x, start_y, end_x - start_x, end_y - start_y, 
                                 thread_counter % num_threads)
          futures.append(future)
          thread_counter += 1
      
      # Wait for all threads to complete
      for future in as_completed(futures):
        future.result()

    # Clean up all readers
    for thread_reader in thread_readers:
      del thread_reader

    return image
    
    
