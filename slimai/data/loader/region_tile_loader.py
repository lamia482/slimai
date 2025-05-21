import cv2
import hashlib
import mmengine
import numpy as np
from typing import Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sdk.reader import get_reader_by_file
from slimai.helper.help_build import LOADERS
from slimai.helper.help_utils import _CACHE_ROOT_DIR_


@LOADERS.register_module()
class RegionTileLoader():
  def __init__(self, *, 
               magnification: int, 
               region: Dict, 
               shrink: str=None,
               cache: bool=True, 
               cache_mode: str="raw", 
               num_threads: int=None, 
               padding_value: int=255,
               ):
    self.magnification = magnification
    self.region = region
    assert (
      region is None or (
        isinstance(region, dict) and {"xmin", "ymin", "xmax", "ymax"}.issubset(set(region.keys()))
      )
    ), "Region must be None or a dictionary with keys: 'xmin', 'ymin', 'xmax', 'ymax'"
    self.shrink = shrink
    self.cache = cache
    assert (
      cache_mode in ["raw", "compressed"]
    ), "cache_mode is expected to be one of ['raw', 'compressed'], but got: {}".format(cache_mode)
    self.compressed = False if cache_mode == "raw" else True
    self.num_threads = num_threads
    self.padding_value = padding_value
    return
  
  def __call__(self, file):
    cache_file = Path(_CACHE_ROOT_DIR_, "loader", self.__class__.__name__, "{}-{}.pkl".format(
      hashlib.md5("+".join(map(str, [self.magnification, self.region, self.padding_value])
      ).encode(encoding="UTF-8")).hexdigest(), 
      hashlib.md5(file.encode(encoding="UTF-8")).hexdigest() + ("" if self.compressed == "raw" else "-compressed")
    ))

    if self.cache and cache_file.exists():
      try:
        data = mmengine.load(cache_file)
        if self.compressed:
          data = cv2.imdecode(np.frombuffer(data, "uint8"), cv2.IMREAD_COLOR)
        return data
      except Exception as e:
        pass
    
    wsi_file_path = file

    reader = get_reader_by_file(wsi_file_path, scale=self.magnification)
    if not reader.status:
      return None
    
    xmin, ymin, xmax, ymax = [self.region[k] for k in ["xmin", "ymin", "xmax", "ymax"]]
    if 0 < xmin < 1:
      xmin = int(xmin * reader.getReadWidth())
    if 0 < ymin < 1:
      ymin = int(ymin * reader.getReadHeight())
    if 0 < xmax < 1:
      xmax = int(xmax * reader.getReadWidth())
    if 0 < ymax < 1:
      ymax = int(ymax * reader.getReadHeight())
    if xmax <= 0:
      xmax = reader.getReadWidth()
    if ymax <= 0:
      ymax = reader.getReadHeight()
    
    if self.num_threads:
      tile = self.read_roi_async(reader, xmin, ymin, xmax, ymax, self.num_threads)
    else:
      tile = reader.ReadRoi(xmin, ymin, xmax-xmin, ymax-ymin, scale=reader.getReadScale())

    if self.cache:
      data = tile
      if self.compressed:
        data = cv2.imencode(".jpg", data)[1].tobytes()
      mmengine.dump(data, cache_file)
    
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
      def read_roi_chunk(ox, oy, thread_id, x, y, w, h):
        if w <= 0 or h <= 0:
          return
        # Get reader from thread's reader list
        thread_reader = thread_readers[thread_id]
        try:
          chunk = thread_reader.ReadRoi(x, y, w, h, scale=thread_reader.getReadScale())
        except Exception as e:
          chunk = np.full((h, w, 3), self.padding_value, dtype=np.uint8)
        x, y = x - ox, y - oy
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
          
          future = executor.submit(read_roi_chunk, xmin, ymin, thread_counter % num_threads, 
                                   start_x, start_y, end_x - start_x, end_y - start_y)
          futures.append(future)
          thread_counter += 1
      
      # Wait for all threads to complete
      for future in as_completed(futures):
        future.result()

    # Clean up all readers
    for thread_reader in thread_readers:
      del thread_reader

    return image
    
    
