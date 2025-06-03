import os
import hashlib
import mmengine
import numpy as np
from pathlib import Path
from sdk.reader import get_reader_by_file
from slimai.helper.help_build import LOADERS
from slimai.helper.common import CACHE_ROOT_DIR
from concurrent.futures import ThreadPoolExecutor, as_completed


@LOADERS.register_module()
class StackTileLoader():
  """
  Stack tiles from a WSI file, output shape: (N, H, W, C)
  """
  def __init__(self, *, 
               magnification, 
               tile_size,
               tile_stride,
               region=None, 
               padding_value=255, 
               num_threads=None, 
               cache=True, 
               ):
    self.magnification = magnification
    self.tile_size = tile_size
    self.tile_stride = tile_stride
    self.region = region
    self.padding_value = padding_value
    self.num_threads = num_threads
    self.cache = cache
    return
  
  def __call__(self, file):
    cache_file = Path(CACHE_ROOT_DIR, "loader", self.__class__.__name__, "{}-{}.pkl".format(
      hashlib.md5("+".join(map(str, [self.magnification, self.tile_size, self.tile_stride, self.region, self.padding_value])
      ).encode(encoding="UTF-8")).hexdigest(), 
      hashlib.md5(file.encode(encoding="UTF-8")).hexdigest()
    ))
    if self.cache and cache_file.exists():
      try:
        return mmengine.load(cache_file)
      except Exception as e:
        pass

    wsi_file_path = file

    reader = get_reader_by_file(wsi_file_path, scale=self.magnification)
    if not reader.status:
      return None
    
    if self.region is not None:
      x, y, w, h = self.region
    else:
      x, y, w, h = 0, 0, reader.getReadWidth(), reader.getReadHeight()
        
    image_list = self._read_roi_async(reader, x, y, w, h)

    if self.cache:
      mmengine.dump(image_list, cache_file)
    
    return image_list

  def _read_roi_async(self, reader, xmin, ymin, xmax, ymax):
    def read_roi_chunk(start_x, end_x, start_y, end_y, thread_id):
      # Get reader from thread's reader list
      thread_reader = thread_readers[thread_id]
      try:
        chunk = thread_reader.ReadRoi(start_x, start_y, end_x - start_x, end_y - start_y, scale=thread_reader.getReadScale())
        # Pad if needed
        if chunk.shape[0] < self.tile_size or chunk.shape[1] < self.tile_size:
          padded = np.full((self.tile_size, self.tile_size, 3), self.padding_value, dtype=np.uint8)
          padded[:chunk.shape[0], :chunk.shape[1], :] = chunk
          chunk = padded
      except Exception as e:
        chunk = np.full((self.tile_size, self.tile_size, 3), self.padding_value, dtype=np.uint8)
      return chunk
    
    # Calculate number of tiles needed
    y_chunks = (ymax - ymin + self.tile_size - 1) // self.tile_size
    x_chunks = (xmax - xmin + self.tile_size - 1) // self.tile_size

    num_threads = os.cpu_count() or 1
    num_threads = self.num_threads or min(num_threads, x_chunks * y_chunks) // 4

    # Create readers upfront for each thread
    file_path, scale = reader.file, reader.getReadScale()
    thread_readers = []
    for _ in range(num_threads):
      thread_reader = get_reader_by_file(file_path, scale=scale)
      thread_readers.append(thread_reader)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
      futures = []
      
      # Create grid of chunks
      thread_counter = 0
      for i in range(y_chunks):
        start_y = ymin + i * self.tile_size
        end_y = min(start_y + self.tile_size, ymax)
        
        for j in range(x_chunks):
          start_x = xmin + j * self.tile_size
          end_x = min(start_x + self.tile_size, xmax)
          
          future = executor.submit(read_roi_chunk, start_x, end_x, start_y, end_y, 
                                 thread_counter % num_threads)
          futures.append(future)
          thread_counter += 1
      
      # Wait for all threads to complete
      image_list = []
      for future in as_completed(futures):
        image_list.append(future.result())

    # Clean up all readers
    for thread_reader in thread_readers:
      del thread_reader

    return image_list
    