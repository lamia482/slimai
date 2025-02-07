import numpy as np
import mmengine
from sdk.reader import get_reader_by_file
from slimai.helper.help_build import LOADERS
from slimai.helper.help_utils import print_log


@LOADERS.register_module()
class ReadWsiLoader():
  def __init__(self, *, 
               random_scale=[10, 20, 20, 20, 40], 
               random_crop_size=[256, 512, 1024], 
               anchor_file=None
               ):
    self.random_scale = random_scale
    self.random_crop_size = random_crop_size
    self.anchors = None if anchor_file is None else mmengine.load(anchor_file)
    return
  
  def __call__(self, file):
    wsi_file_path = file

    random_read_scale = int(np.random.choice(self.random_scale))
    random_crop_size = int(np.random.choice(self.random_crop_size))
    reader = get_reader_by_file(wsi_file_path, scale=random_read_scale)
    if not reader.status:
      return None
    x, y, w, h = self.get_random_crop_position(reader.getSrcWidth(), reader.getSrcHeight(), random_crop_size, self.anchors)
    image = reader.ReadRoi(x, y, w, h, scale=reader.getReadScale())
    return image
  
  @classmethod
  def get_random_crop_position(cls, src_width, src_height, crop_size, anchors=None):
    if anchors is None:
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
    else:
      x, y = np.random.choice(anchors)
    return x, y, crop_size, crop_size
