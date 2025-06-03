import numpy as np
from sdk.reader import get_reader_by_file
from slimai.helper.help_build import LOADERS


@LOADERS.register_module()
class RandomTileLoader():
  def __init__(self, *, 
               random_scale=[10, 20, 20, 20, 40], 
               random_crop_size=[256, 512, 1024], 
               ):
    self.random_scale = random_scale
    self.random_crop_size = random_crop_size
    return
  
  def __call__(self, file):
    wsi_file_path = file

    random_read_scale = int(np.random.choice(self.random_scale))
    random_crop_size = int(np.random.choice(self.random_crop_size))
    reader = get_reader_by_file(wsi_file_path, scale=random_read_scale)
    if not reader.status:
      return None
    x, y, w, h = self.get_random_crop_position(reader.getReadWidth(), reader.getReadHeight(), random_crop_size)
    image = reader.ReadRoi(x, y, w, h, scale=reader.getReadScale()) # type: ignore
    return image
  
  @classmethod
  def get_random_crop_position(cls, src_width, src_height, crop_size):
    # Get random position within the circle
    radius = crop_size // 2
    center_x = src_width // 2
    center_y = src_height // 2
    
    # Generate random angle and distance from center
    angle = np.random.uniform(0, 2 * np.pi)
    # Use sqrt for uniform distribution within circle
    distance = radius * np.sqrt(np.random.uniform(0, 1))
    
    # Convert polar to cartesian coordinates
    x = int(center_x + distance * np.cos(angle))
    y = int(center_y + distance * np.sin(angle))
    
    # Ensure coordinates stay within image bounds
    x = max(radius, min(x, src_width - radius))
    y = max(radius, min(y, src_height - radius))

    return x, y, crop_size, crop_size
