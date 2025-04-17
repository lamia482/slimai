
__all__ = ["singleton_wrapper"]


def singleton_wrapper(class_):
  """A decorator to make a class a singleton.
  
  Args:
    class_: The class to make a singleton.
  
  Returns:
    The singleton instance of the class.
  """
  instances = {}
  def getinstance(*args, **kwargs):
    if class_ not in instances:
      instances[class_] = class_(*args, **kwargs)
    return instances[class_]
  return getinstance
