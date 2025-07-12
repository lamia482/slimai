import threading


__all__ = [
  "singleton_wrapper",
  "classproperty",
] 

def singleton_wrapper(class_):
  """A decorator to make a class a singleton.
  
  Args:
    class_: The class to make a singleton.
  
  Returns:
    The singleton instance of the class.
  """
  instances = {}
  lock = threading.RLock()
  
  def getinstance(*args, **kwargs):
    if class_ not in instances:
      with lock:
        # Double-check locking pattern
        if class_ not in instances:
          instances[class_] = class_(*args, **kwargs)
    return instances[class_]
  
  return getinstance


class classproperty:
  def __init__(self, method):
    self.method = method
    return

  def __get__(self, instance, cls=None):
    return self.method(cls)
