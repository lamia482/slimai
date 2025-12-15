import concurrent.futures
from multiprocessing.pool import ThreadPool, Pool as ProcessPool


_EXPECTED_ASYNC_TYPE = ["thread", "process"]
_DEFAULT_ASYNC_TYPE = "thread"

def submit_parallel_task(func, args_list, async_type: str = _DEFAULT_ASYNC_TYPE, max_workers: int = 1):
  assert (
    async_type in _EXPECTED_ASYNC_TYPE
  ), f"Invalid async type: {async_type}, expected one of: {_EXPECTED_ASYNC_TYPE}"

  assert (
    isinstance(args_list, list)
  ), f"args_list must be a list, but got {type(args_list)}"
  
  if async_type == "thread":
    Executor = concurrent.futures.ThreadPoolExecutor
  elif async_type == "process":
    Executor = concurrent.futures.ProcessPoolExecutor
  
  with Executor(max_workers=max_workers) as executor:
    futures = executor.map(func, args_list)
    return list(futures)

def submit_async_task(func, *args, async_type: str = _DEFAULT_ASYNC_TYPE):
  assert (
    async_type in _EXPECTED_ASYNC_TYPE
  ), f"Invalid async type: {async_type}, expected one of: {_EXPECTED_ASYNC_TYPE}"
  Executor = ThreadPool if async_type == "thread" else ProcessPool
  # Use context manager for resource safety
  executor = Executor(processes=1)
  try:
    future = executor.apply_async(func, args=args)
    # Provide a simple interface for user to check completion and get result
    class AsyncTaskHandle:
      def __init__(self, future, executor):
        self._future = future
        self._executor = executor
      def ready(self):
        return self._future.ready()
      def wait(self, timeout=None):
        return self._future.wait(timeout=timeout)
      def get(self, timeout=None):
        result = self._future.get(timeout=timeout)
        self._executor.close()
        self._executor.join()
        return result
    return AsyncTaskHandle(future, executor)
  except Exception as e:
    executor.terminate()
    executor.join()
    raise e
