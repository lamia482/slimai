from __future__ import annotations

import time
from typing import Any, Dict, Iterator, Optional, TypeVar

T = TypeVar("T")


def validation_progress(
  iterable,
  *,
  total: Optional[int] = None,
  desc: str,
  unit: str = "it",
  enabled: bool = True,
  leave: bool = False,
  position: int = 0,
):
  if not enabled:
    return iterable
  try:
    from tqdm import tqdm

    return tqdm(
      iterable,
      total=total,
      desc=desc,
      unit=unit,
      leave=leave,
      position=position,
    )
  except Exception:
    return iterable


class ValidationPhaseTimer:
  def __init__(self, timing: Dict[str, Any], phase: str):
    self.timing = timing
    self.phase = phase
    self._start = 0.0
    return

  def __enter__(self):
    self._start = time.perf_counter()
    return self

  def __exit__(self, exc_type, exc, tb):
    elapsed = time.perf_counter() - self._start
    phases = self.timing.setdefault("phases", {})
    phases[self.phase] = round(float(elapsed), 4)
    return False


class TopLevelValidationProgress:
  def __init__(self, *, enabled: bool, total: int = 8):
    self.enabled = enabled
    self.total = total
    self._bar = None
    return

  def __enter__(self):
    if not self.enabled:
      return self
    try:
      from tqdm import tqdm

      self._bar = tqdm(total=self.total, desc="Export validation", unit="phase", leave=True)
    except Exception:
      self._bar = None
    return self

  def __exit__(self, exc_type, exc, tb):
    if self._bar is not None:
      self._bar.close()
    return False

  def update(self, n: int = 1):
    if self._bar is not None:
      self._bar.update(n)
    return


class PostfixProgress:
  def __init__(self, bar):
    self._bar = bar
    return

  def __iter__(self) -> Iterator[T]:
    return iter(self._bar)

  def set_postfix(self, **kwargs):
    if hasattr(self._bar, "set_postfix"):
      self._bar.set_postfix(**kwargs)
    return


def iter_with_postfix(
  iterable,
  *,
  enabled: bool,
  desc: str,
  unit: str,
  total: Optional[int] = None,
):
  bar = validation_progress(iterable, total=total, desc=desc, unit=unit, enabled=enabled, leave=False)
  try:
    from tqdm import tqdm

    if enabled and isinstance(bar, tqdm):
      return PostfixProgress(bar)
  except Exception:
    pass
  return bar
