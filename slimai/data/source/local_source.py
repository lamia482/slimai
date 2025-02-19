import glob
import os.path as osp
from typing import Dict, List
from slimai.helper.help_build import SOURCES


@SOURCES.register_module()
class LocalSource(object):
  def __init__(self, path, ext=None):
    self.path = path
    self.ext = ext
    return

  def __call__(self) -> Dict[str, List[str]]:
    ext = self.ext
    if ext is None:
      ext = ""
    if not isinstance(ext, (tuple, list)):
      ext = [ext]
    files = []
    for e in ext:
      if not e.startswith("."):
        e = f".{e}"
      files.extend(self.get_files(e))
    return dict(files=files)
    
  def get_files(self, ext):
    return glob.glob(osp.join(self.path, "**", f"*{ext}"), recursive=True)
