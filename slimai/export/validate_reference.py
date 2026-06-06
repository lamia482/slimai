from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_float(value: str) -> Optional[float]:
  text = str(value).strip()
  if text in {"", "N/A", "nan", "None"}:
    return None
  try:
    return float(text)
  except Exception:
    return None


class _CompareTableParser(HTMLParser):
  def __init__(self):
    super().__init__()
    self.tables: List[List[List[str]]] = []
    self._in_table = False
    self._in_row = False
    self._in_cell = False
    self._cell_tag = ""
    self._row: List[str] = []
    self._cell_parts: List[str] = []
    self._current_table: List[List[str]] = []
    return

  def handle_starttag(self, tag, attrs):
    if tag == "table":
      self._in_table = True
      self._current_table = []
    elif self._in_table and tag == "tr":
      self._in_row = True
      self._row = []
    elif self._in_row and tag in {"th", "td"}:
      self._in_cell = True
      self._cell_tag = tag
      self._cell_parts = []
    return

  def handle_endtag(self, tag):
    if tag in {"th", "td"} and self._in_cell:
      self._row.append("".join(self._cell_parts).strip())
      self._in_cell = False
    elif tag == "tr" and self._in_row:
      if len(self._row) > 0:
        self._current_table.append(self._row)
      self._in_row = False
    elif tag == "table" and self._in_table:
      if len(self._current_table) > 0:
        self.tables.append(self._current_table)
      self._in_table = False
    return

  def handle_data(self, data):
    if self._in_cell:
      self._cell_parts.append(data)
    return


def _table_to_metric_dict(table: List[List[str]], column_name: str) -> Dict[str, Optional[float]]:
  if len(table) < 2:
    return {}
  header = [cell.strip() for cell in table[0]]
  if column_name not in header:
    return {}
  col_idx = header.index(column_name)
  metrics: Dict[str, Optional[float]] = {}
  for row in table[1:]:
    if len(row) <= col_idx:
      continue
    metric_name = row[0].strip().lower()
    metrics[metric_name] = _parse_float(row[col_idx])
  return metrics


def _parse_external_table(table: List[List[str]]) -> Dict[str, Dict[str, Optional[float]]]:
  if len(table) < 2:
    return {}
  header = [cell.strip().lower() for cell in table[0]]
  if "dataset" not in header:
    return {}
  dataset_idx = header.index("dataset")
  external: Dict[str, Dict[str, Optional[float]]] = {}
  for row in table[1:]:
    if len(row) <= dataset_idx:
      continue
    name = row[dataset_idx].strip()
    if name == "":
      continue
    item: Dict[str, Optional[float]] = {}
    for idx, key in enumerate(header):
      if idx == dataset_idx:
        continue
      if idx >= len(row):
        continue
      item[key] = _parse_float(row[idx])
    external[f"external_{name}"] = item
  return external


def parse_report_html_metrics(html_path: Path, *, inner_column: str = "Inner Test（全量）") -> Dict[str, Any]:
  text = Path(html_path).read_text(encoding="utf-8")
  parser = _CompareTableParser()
  parser.feed(text)
  compare_tables = [
    table for table in parser.tables
    if len(table) > 0 and table[0] and table[0][0].strip().lower() == "metric"
  ]
  inner_test: Dict[str, Optional[float]] = {}
  if len(compare_tables) > 0:
    inner_test = _table_to_metric_dict(compare_tables[0], inner_column)
  external: Dict[str, Dict[str, Optional[float]]] = {}
  for table in parser.tables:
    if len(table) > 0 and table[0] and table[0][0].strip().lower() == "dataset":
      external.update(_parse_external_table(table))
  return dict(inner_test=inner_test, external=external)


def load_experiment_report_baseline(work_dir: Path) -> Dict[str, Any]:
  work_dir = Path(work_dir)
  baseline: Dict[str, Any] = dict(
    work_dir=str(work_dir),
    level1=dict(inner_test={}, external={}),
    level2=dict(inner_test={}, external={}),
    warnings=[],
  )
  level1_path = work_dir / "report_primary.html"
  level2_path = work_dir / "report_marginal.html"
  if not level1_path.exists():
    level1_path = work_dir / "report_level1.html"
  if not level2_path.exists():
    level2_path = work_dir / "report_level2.html"
  if level1_path.exists():
    parsed = parse_report_html_metrics(level1_path)
    baseline["level1"]["inner_test"] = parsed.get("inner_test", {})
    baseline["level1"]["external"] = parsed.get("external", {})
  else:
    baseline["warnings"].append(f"missing: {level1_path}")
  if level2_path.exists():
    parsed = parse_report_html_metrics(level2_path)
    baseline["level2"]["inner_test"] = parsed.get("inner_test", {})
    baseline["level2"]["external"] = parsed.get("external", {})
  else:
    baseline["warnings"].append(f"missing: {level2_path}")
  return baseline


def compare_metric_tables(
  onnx_metrics: Dict[str, Any],
  baseline_metrics: Dict[str, Any],
  *,
  metrics_tol: float,
) -> List[Dict[str, Any]]:
  rows: List[Dict[str, Any]] = []
  keys = sorted(set(list(onnx_metrics.keys()) + list(baseline_metrics.keys())))
  for key in keys:
    if key == "n_samples":
      continue
    onnx_val = onnx_metrics.get(key)
    base_val = baseline_metrics.get(key)
    if onnx_val is None or base_val is None:
      rows.append(dict(metric=key, onnx=onnx_val, original=base_val, delta=None, match=False))
      continue
    delta = float(onnx_val) - float(base_val)
    rows.append(
      dict(
        metric=key,
        onnx=float(onnx_val),
        original=float(base_val),
        delta=delta,
        match=abs(delta) < metrics_tol,
      )
    )
  return rows
