import ast
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mmengine
import numpy as np
import torch
from sklearn.metrics import auc, average_precision_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

from slimai.helper.help_utils import print_log


class ExperimentReporter(object):
  def __init__(
    self,
    *,
    work_dir: Path,
    cfg,
    train_dataloader=None,
    valid_dataloader=None,
    test_dataloader=None,
    model_desc: Optional[str] = None,
  ):
    self.work_dir = Path(work_dir).resolve()
    self.results_dir = self.work_dir / "results"
    self.results_dir.mkdir(parents=True, exist_ok=True)
    self.cfg = cfg
    self.model_desc = model_desc or ""
    self.chart_counter = 0
    self.dataset_info = {
      "train": self._collect_dataset_info(train_dataloader),
      "valid": self._collect_dataset_info(valid_dataloader),
      "test": self._collect_dataset_info(test_dataloader),
    }
    self.class_names = self._resolve_class_names()
    self.display_class_names = self._resolve_display_class_names(self.class_names)
    return

  def _next_chart_id(self, prefix: str):
    self.chart_counter += 1
    return f"{prefix}_{self.chart_counter}"

  def _to_plain(self, value: Any):
    if isinstance(value, torch.Tensor):
      value = value.detach().cpu()
      if value.numel() == 1:
        return float(value.item())
      return value.tolist()
    if isinstance(value, np.ndarray):
      return value.tolist()
    if isinstance(value, (list, tuple)):
      return [self._to_plain(v) for v in value]
    if isinstance(value, dict):
      return {k: self._to_plain(v) for k, v in value.items()}
    return value

  def _format_float(self, value: float):
    return format(float(value), ".8g")

  def _format_plain(self, value: Any):
    value = self._to_plain(value)
    if isinstance(value, bool):
      return str(value)
    if isinstance(value, (np.floating, float)):
      return self._format_float(float(value))
    if isinstance(value, (np.integer, int)):
      return str(int(value))
    if isinstance(value, list):
      return [self._format_plain(v) for v in value]
    if isinstance(value, dict):
      return {k: self._format_plain(v) for k, v in value.items()}
    return value

  def _display_class_name(self, index: int, name: Optional[str] = None):
    text = str(index if name is None else name)
    prefix = f"{index} - "
    if text.startswith(prefix):
      return text
    return f"{index} - {text}"

  def _resolve_display_class_names(self, class_names: List[str]):
    if len(class_names) == 0:
      return []
    return [self._display_class_name(i, class_names[i]) for i in range(len(class_names))]

  def _is_generic_class_names(self, class_names: List[str]):
    if len(class_names) == 0:
      return True
    for i, name in enumerate(class_names):
      if str(name) != str(i):
        return False
    return True

  def _resolve_class_names_from_cfg(self, expected_num_classes: Optional[int] = None):
    loader_keys = ["VALID_LOADER", "VAL_LOADER", "TEST_LOADER", "TRAIN_LOADER"]
    for key in loader_keys:
      loader_cfg = self.cfg.get(key, None)
      if not isinstance(loader_cfg, dict):
        continue
      dataset_cfg = loader_cfg.get("dataset", None)
      if not isinstance(dataset_cfg, dict):
        continue
      label_mapping = dataset_cfg.get("label_mapping", None)
      if isinstance(label_mapping, str):
        try:
          label_mapping = ast.literal_eval(label_mapping)
        except Exception:
          label_mapping = None
      if not isinstance(label_mapping, dict) or len(label_mapping) == 0:
        continue
      inv = {}
      for name, index in label_mapping.items():
        if isinstance(index, (int, np.integer)):
          inv[int(index)] = str(name)
      if len(inv) == 0:
        continue
      max_index = max(inv.keys())
      num_classes = max_index + 1
      if isinstance(expected_num_classes, int):
        num_classes = max(num_classes, expected_num_classes)
      return [inv.get(i, str(i)) for i in range(num_classes)]
    return None

  def _is_long_text(self, text: str):
    return (len(text) > 48) or ("/" in text) or ("\\" in text)

  def _render_value_html(self, value: Any):
    value = self._format_plain(value)
    if isinstance(value, (dict, list)):
      json_text = json.dumps(value, ensure_ascii=False, indent=2)
      return (
        "<div class='scroll-x'>"
        f"<pre>{html.escape(json_text)}</pre>"
        "</div>"
      )
    text = html.escape(str(value))
    if self._is_long_text(str(value)):
      return f"<div class='scroll-x'>{text}</div>"
    return text

  def _collect_dataset_info(self, dataloader):
    if dataloader is None:
      return None
    dataset = dataloader.dataset
    info = dict(
      size=len(dataset),
      desc=getattr(dataset, "desc", str(dataset)),
      class_names=list(getattr(dataset, "class_names", [])),
      split_stat=self._to_plain(getattr(dataset, "split_stat", None)),
      split_file=getattr(dataset, "split_file", None),
    )

    labels = None
    if hasattr(dataset, "annotations") and isinstance(dataset.annotations, dict):
      labels = dataset.annotations.get("label", None)

    if labels is not None:
      class_names = info["class_names"]
      if len(class_names) == 0:
        max_label = int(max(labels)) if len(labels) > 0 else -1
        class_names = [str(i) for i in range(max_label + 1)]
      if self._is_generic_class_names(class_names):
        fallback_names = self._resolve_class_names_from_cfg(expected_num_classes=len(class_names))
        if isinstance(fallback_names, list) and len(fallback_names) > 0:
          class_names = fallback_names
      display_names = [self._display_class_name(i, name) for i, name in enumerate(class_names)]
      raw_dist = {name: 0 for name in display_names}
      for label in labels:
        label = int(label)
        if 0 <= label < len(display_names):
          raw_dist[display_names[label]] += 1
      sampled_dist = {name: 0 for name in display_names}
      indices = list(getattr(dataset, "indices", list(range(len(labels)))))
      for index in indices:
        label = int(labels[index])
        if 0 <= label < len(display_names):
          sampled_dist[display_names[label]] += 1
      info["raw_label_distribution"] = raw_dist
      info["sampled_label_distribution"] = sampled_dist
    return info

  def _resolve_class_names(self):
    for key in ["valid", "test", "train"]:
      info = self.dataset_info.get(key, None)
      if info is not None and len(info.get("class_names", [])) > 0:
        class_names = info["class_names"]
        if not self._is_generic_class_names(class_names):
          return class_names
    cfg_class_names = self._resolve_class_names_from_cfg()
    if isinstance(cfg_class_names, list) and len(cfg_class_names) > 0:
      return cfg_class_names
    for key in ["valid", "test", "train"]:
      info = self.dataset_info.get(key, None)
      if info is not None and len(info.get("class_names", [])) > 0:
        return info["class_names"]
    metric_cfg = self.cfg.get("METRIC", dict())
    num_classes = (
      metric_cfg.get("acc", dict()).get("num_classes", None)
      or metric_cfg.get("auc", dict()).get("num_classes", None)
      or metric_cfg.get("f1", dict()).get("num_classes", None)
    )
    if num_classes is None:
      return []
    return [str(i) for i in range(int(num_classes))]

  def _extract_eval_arrays(self, result_file: Path):
    if not Path(result_file).exists():
      return None, None, None
    result = mmengine.load(result_file)
    batch_info = result.get("batch_info", [])
    labels = []
    preds = []
    probs = []
    for item in batch_info:
      label = item.get("label", None) if isinstance(item, dict) else getattr(item, "label", None)
      output = item.get("output", None) if isinstance(item, dict) else getattr(item, "output", None)
      if output is None:
        continue
      softmax = output.get("softmax", None) if isinstance(output, dict) else getattr(output, "softmax", None)
      pred = output.get("labels", None) if isinstance(output, dict) else getattr(output, "labels", None)
      if label is None or softmax is None or pred is None:
        continue
      label = int(torch.as_tensor(label).cpu().item())
      pred = int(torch.as_tensor(pred).cpu().item())
      softmax = torch.as_tensor(softmax).detach().cpu().float().numpy()
      labels.append(label)
      preds.append(pred)
      probs.append(softmax)
    if len(labels) == 0:
      return None, None, None
    return np.asarray(labels), np.asarray(preds), np.asarray(probs)

  def _chart_svg_line(
    self,
    *,
    chart_id: str,
    title: str,
    x_label: str,
    y_label: str,
    x_values: List[float],
    series_list: List[Dict[str, Any]],
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
  ):
    if len(x_values) == 0 or len(series_list) == 0:
      return "<p>N/A</p>"

    width = 860
    height = 430
    margin_left = 76
    margin_right = 20
    margin_top = 42
    margin_bottom = 58
    inner_w = width - margin_left - margin_right
    inner_h = height - margin_top - margin_bottom

    xs = [float(v) for v in x_values]
    all_y = []
    for series in series_list:
      for v in series["y"]:
        if v is None:
          continue
        all_y.append(float(v))
    if len(all_y) == 0:
      return "<p>N/A</p>"

    if x_range is None:
      x_min, x_max = min(xs), max(xs)
      if x_min == x_max:
        x_max = x_min + 1.0
    else:
      x_min, x_max = x_range
    if y_range is None:
      y_min, y_max = min(all_y), max(all_y)
      pad = max((y_max - y_min) * 0.08, 1e-4)
      y_min -= pad
      y_max += pad
      if y_min == y_max:
        y_max = y_min + 1.0
    else:
      y_min, y_max = y_range

    def _sx(value: float):
      return margin_left + (value - x_min) / (x_max - x_min) * inner_w

    def _sy(value: float):
      return margin_top + (1.0 - (value - y_min) / (y_max - y_min)) * inner_h

    grid_lines = []
    for i in range(6):
      x = margin_left + i * inner_w / 5.0
      y = margin_top + i * inner_h / 5.0
      grid_lines.append(f"<line x1='{x:.2f}' y1='{margin_top:.2f}' x2='{x:.2f}' y2='{margin_top + inner_h:.2f}' class='grid-line' />")
      grid_lines.append(f"<line x1='{margin_left:.2f}' y1='{y:.2f}' x2='{margin_left + inner_w:.2f}' y2='{y:.2f}' class='grid-line' />")

    x_ticks = []
    for i in range(6):
      value = x_min + (x_max - x_min) * i / 5.0
      x = _sx(value)
      label = self._format_float(value)
      x_ticks.append(f"<text x='{x:.2f}' y='{height - 28}' text-anchor='middle' class='axis-tick'>{html.escape(label)}</text>")
    y_ticks = []
    for i in range(6):
      value = y_min + (y_max - y_min) * (5 - i) / 5.0
      y = margin_top + i * inner_h / 5.0
      label = self._format_float(value)
      y_ticks.append(f"<text x='{margin_left - 8}' y='{y + 4:.2f}' text-anchor='end' class='axis-tick'>{html.escape(label)}</text>")

    line_blocks = []
    legend_buttons = []
    for i, series in enumerate(series_list):
      points = []
      circles = []
      for x_val, y_val in zip(xs, series["y"]):
        if y_val is None:
          continue
        px = _sx(float(x_val))
        py = _sy(float(y_val))
        points.append(f"{px:.2f},{py:.2f}")
        circles.append(f"<circle cx='{px:.2f}' cy='{py:.2f}' r='2.6' fill='{series['color']}' />")
      if len(points) == 0:
        continue
      series_id = f"{chart_id}_s{i}"
      line_blocks.append(
        f"<g class='series-group' data-series-id='{series_id}'>"
        f"<polyline points='{' '.join(points)}' fill='none' stroke='{series['color']}' stroke-width='2.4' stroke-linejoin='round' stroke-linecap='round' />"
        f"{''.join(circles)}"
        "</g>"
      )
      legend_buttons.append(
        f"<button class='legend-item active' type='button' data-series-target='{series_id}'>"
        f"<span class='legend-dot' style='background:{series['color']}'></span>"
        f"{html.escape(str(series['name']))}"
        "</button>"
      )

    if len(line_blocks) == 0:
      return "<p>N/A</p>"

    svg = (
      f"<svg class='line-chart-svg' viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>"
      f"<text x='{width/2:.2f}' y='24' text-anchor='middle' class='chart-title'>{html.escape(title)}</text>"
      f"{''.join(grid_lines)}"
      f"<line x1='{margin_left}' y1='{margin_top + inner_h}' x2='{margin_left + inner_w}' y2='{margin_top + inner_h}' class='axis-line' />"
      f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{margin_top + inner_h}' class='axis-line' />"
      f"{''.join(x_ticks)}"
      f"{''.join(y_ticks)}"
      f"<text x='{width/2:.2f}' y='{height - 8}' text-anchor='middle' class='axis-label'>{html.escape(x_label)}</text>"
      f"<text x='18' y='{height/2:.2f}' text-anchor='middle' transform='rotate(-90 18 {height/2:.2f})' class='axis-label'>{html.escape(y_label)}</text>"
      f"{''.join(line_blocks)}"
      "</svg>"
    )
    return (
      f"<div class='chart-shell interactive-chart' id='{chart_id}'>"
      f"<div class='legend-row'>{''.join(legend_buttons)}</div>"
      f"{svg}"
      "<script>"
      "(function(){"
      f"const root=document.getElementById('{chart_id}');"
      "if(!root){return;}"
      "root.querySelectorAll('.legend-item').forEach(function(btn){"
      "btn.addEventListener('click',function(){"
      "const target=btn.getAttribute('data-series-target');"
      "const line=root.querySelector(\".series-group[data-series-id='\"+target+\"']\");"
      "if(!line){return;}"
      "const active=btn.classList.toggle('active');"
      "line.style.display=active?'':'none';"
      "});"
      "});"
      "})();"
      "</script>"
      "</div>"
    )

  def _chart_confusion_matrix(self, labels, preds, class_names):
    if len(labels) == 0:
      return "<p>N/A</p>"
    n_classes = max(int(max(labels)), int(max(preds))) + 1
    display_names = []
    for i in range(n_classes):
      if i < len(class_names):
        display_names.append(class_names[i])
      else:
        display_names.append(self._display_class_name(i))
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    max_value = int(cm.max()) if cm.size > 0 else 1
    if max_value <= 0:
      max_value = 1

    rows = []
    for i in range(n_classes):
      cells = [f"<th>{html.escape(display_names[i])}</th>"]
      for j in range(n_classes):
        value = int(cm[i, j])
        alpha = value / max_value
        bg = f"rgba(37, 99, 235, {0.10 + alpha * 0.82:.4f})"
        text_color = "#ffffff" if alpha > 0.55 else "#0f172a"
        cells.append(
          f"<td style='background:{bg};color:{text_color};'>{value}</td>"
        )
      rows.append("<tr>" + "".join(cells) + "</tr>")
    head_cells = "".join([f"<th>{html.escape(name)}</th>" for name in display_names])
    return (
      "<div class='cm-wrap'>"
      "<div class='cm-title'>Confusion Matrix</div>"
      "<div class='scroll-x'>"
      "<table class='cm-table'>"
      f"<tr><th>True \\ Pred</th>{head_cells}</tr>"
      f"{''.join(rows)}"
      "</table>"
      "</div>"
      "</div>"
    )

  def _plot_roc_curve(self, labels, probs, class_names):
    if probs.ndim != 2 or probs.shape[0] == 0:
      return None
    num_classes = probs.shape[1]
    y_bin = label_binarize(labels, classes=list(range(num_classes)))
    palette = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#0ea5e9"]
    fpr_grid = np.linspace(0.0, 1.0, 120)
    series = []
    for i in range(num_classes):
      y_true = y_bin[:, i]
      if len(np.unique(y_true)) < 2:
        continue
      fpr, tpr, _ = roc_curve(y_true, probs[:, i])
      roc_auc = auc(fpr, tpr)
      name = class_names[i] if i < len(class_names) else self._display_class_name(i)
      y_resampled = self._resample_xy(fpr.tolist(), tpr.tolist(), fpr_grid)
      series.append(
        dict(
          name=f"{name} (AUC={roc_auc:.4f})",
          color=palette[i % len(palette)],
          y=y_resampled,
        )
      )
    if len(series) == 0:
      return None
    chart_id = self._next_chart_id("roc")
    return self._chart_svg_line(
      chart_id=chart_id,
      title="ROC Curve (One-vs-Rest)",
      x_label="False Positive Rate",
      y_label="True Positive Rate",
      x_values=fpr_grid.tolist(),
      series_list=series,
      x_range=(0.0, 1.0),
      y_range=(0.0, 1.0),
    )

  def _resample_xy(self, x_values: List[float], y_values: List[float], grid: np.ndarray):
    if len(x_values) == 0:
      return [None for _ in grid]
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x_unique, idx = np.unique(x, return_index=True)
    y_unique = y[idx]
    if len(x_unique) == 1:
      return [float(y_unique[0]) for _ in grid]
    return np.interp(grid, x_unique, y_unique).tolist()

  def _plot_pr_curve(self, labels, probs, class_names):
    if probs.ndim != 2 or probs.shape[0] == 0:
      return None
    num_classes = probs.shape[1]
    y_bin = label_binarize(labels, classes=list(range(num_classes)))
    palette = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#0ea5e9"]
    recall_grid = np.linspace(0.0, 1.0, 120)
    series = []
    for i in range(num_classes):
      y_true = y_bin[:, i]
      if len(np.unique(y_true)) < 2:
        continue
      precision, recall, _ = precision_recall_curve(y_true, probs[:, i])
      ap = average_precision_score(y_true, probs[:, i])
      name = class_names[i] if i < len(class_names) else self._display_class_name(i)
      y_resampled = self._resample_xy(recall.tolist(), precision.tolist(), recall_grid)
      series.append(
        dict(
          name=f"{name} (AP={ap:.4f})",
          color=palette[i % len(palette)],
          y=y_resampled,
        )
      )
    if len(series) == 0:
      return None
    chart_id = self._next_chart_id("pr")
    return self._chart_svg_line(
      chart_id=chart_id,
      title="PR Curve (One-vs-Rest)",
      x_label="Recall",
      y_label="Precision",
      x_values=recall_grid.tolist(),
      series_list=series,
      x_range=(0.0, 1.0),
      y_range=(0.0, 1.0),
    )

  def build_eval_figures(self, *, result_file: Path, file_prefix: str):
    _ = file_prefix
    labels, preds, probs = self._extract_eval_arrays(result_file)
    if labels is None or preds is None or probs is None:
      return {}
    output = dict()
    if roc_html := self._plot_roc_curve(labels, probs, self.display_class_names):
      output["roc"] = roc_html
    if pr_html := self._plot_pr_curve(labels, probs, self.display_class_names):
      output["pr"] = pr_html
    output["cm"] = self._chart_confusion_matrix(labels, preds, self.display_class_names)
    return output

  def _render_dict_as_table(self, data: Dict[str, Any]):
    if data is None:
      return "<p>N/A</p>"
    rows = []
    for key, value in data.items():
      rows.append(
        "<tr>"
        f"<th>{html.escape(str(key))}</th>"
        f"<td>{self._render_value_html(value)}</td>"
        "</tr>"
      )
    return "<table class='flat-table'>" + "".join(rows) + "</table>"

  def _json_block(self, data):
    data = self._format_plain(data)
    json_text = json.dumps(data, ensure_ascii=False, indent=2)
    return f"<div class='scroll-x'><pre>{html.escape(json_text)}</pre></div>"

  def _render_dataset_section(self):
    split_cards = []
    for split in ["train", "valid", "test"]:
      info = self.dataset_info.get(split, None)
      if info is None:
        continue
      split_cards.append(
        "<div class='split-card'>"
        f"<h4>{split.upper()}</h4>"
        f"<p><span class='k'>size</span><span class='v'>{self._render_value_html(info.get('size', 'N/A'))}</span></p>"
        f"<p><span class='k'>desc</span><span class='v'>{self._render_value_html(info.get('desc', 'N/A'))}</span></p>"
        f"<p><span class='k'>split_file</span><span class='v'>{self._render_value_html(info.get('split_file', 'N/A'))}</span></p>"
        f"<p><span class='k'>split_stat</span><span class='v'>{self._render_value_html(info.get('split_stat', 'N/A'))}</span></p>"
        "</div>"
      )

    labels = self.display_class_names
    if len(labels) == 0:
      labels = sorted(
        set(
          list((self.dataset_info.get("train") or {}).get("raw_label_distribution", {}).keys())
          + list((self.dataset_info.get("valid") or {}).get("raw_label_distribution", {}).keys())
          + list((self.dataset_info.get("test") or {}).get("raw_label_distribution", {}).keys())
        )
      )
    rows = []
    for label in labels:
      cols = [f"<td>{html.escape(label)}</td>"]
      for split in ["train", "valid", "test"]:
        info = self.dataset_info.get(split, {}) or {}
        raw = info.get("raw_label_distribution", {}) or {}
        sampled = info.get("sampled_label_distribution", {}) or {}
        raw_total = sum(raw.values()) if len(raw) > 0 else 0
        sampled_total = sum(sampled.values()) if len(sampled) > 0 else 0
        raw_count = int(raw.get(label, 0))
        sampled_count = int(sampled.get(label, 0))
        raw_ratio = (100.0 * raw_count / raw_total) if raw_total > 0 else 0.0
        sampled_ratio = (100.0 * sampled_count / sampled_total) if sampled_total > 0 else 0.0
        cols.append(
          "<td>"
          f"<div class='ratio-line'><b>raw</b>: {raw_count} ({self._format_float(raw_ratio)}%)</div>"
          f"<div class='ratio-line'><b>sampled</b>: {sampled_count} ({self._format_float(sampled_ratio)}%)</div>"
          "</td>"
        )
      rows.append("<tr>" + "".join(cols) + "</tr>")
    compare_table = (
      "<table class='compare-table'>"
      "<tr><th>label</th><th>train</th><th>valid</th><th>test</th></tr>"
      + "".join(rows)
      + "</table>"
    )
    return (
      "<div class='dataset-grid'>"
      f"{''.join(split_cards)}"
      "</div>"
      "<div class='scroll-x'>"
      f"{compare_table}"
      "</div>"
    )

  def _render_epoch_table(self, epoch_records: List[Dict[str, Any]]):
    rows = []
    for item in sorted(epoch_records, key=lambda x: x.get("epoch", -1)):
      rows.append(
        "<tr>"
        f"<td>{self._render_value_html(item.get('epoch', 'N/A'))}</td>"
        f"<td>{self._render_value_html(item.get('valid_loss', 'N/A'))}</td>"
        f"<td>{self._render_value_html(item.get('metrics', {}).get('acc', 'N/A'))}</td>"
        f"<td>{self._render_value_html(item.get('metrics', {}).get('auc', 'N/A'))}</td>"
        f"<td>{self._render_value_html(item.get('metrics', {}).get('kappa', 'N/A'))}</td>"
        f"<td>{self._render_value_html(item.get('metrics', {}).get('f1', 'N/A'))}</td>"
        f"<td>{self._render_value_html(item.get('checkpoint', 'N/A'))}</td>"
        "</tr>"
      )
    return (
      "<table class='flat-table'>"
      "<tr><th>epoch</th><th>valid_loss</th><th>acc</th><th>auc</th><th>kappa</th><th>f1</th><th>checkpoint</th></tr>"
      + "".join(rows)
      + "</table>"
    )

  def _build_valid_loss_chart(self, epoch_records: List[Dict[str, Any]]):
    points = []
    for item in sorted(epoch_records, key=lambda x: x.get("epoch", -1)):
      epoch = item.get("epoch", None)
      loss = item.get("valid_loss", None)
      if isinstance(epoch, int) and isinstance(loss, (int, float)):
        points.append((epoch, float(loss)))
    if len(points) == 0:
      return None
    chart_id = self._next_chart_id("valid_loss")
    return self._chart_svg_line(
      chart_id=chart_id,
      title="Valid Loss Trend",
      x_label="Epoch",
      y_label="Valid Loss",
      x_values=[float(p[0]) for p in points],
      series_list=[dict(name="valid_loss", color="#2563eb", y=[p[1] for p in points])],
    )

  def _build_epoch_metric_chart(self, epoch_records: List[Dict[str, Any]]):
    points = sorted(epoch_records, key=lambda x: x.get("epoch", -1))
    epochs = [int(v["epoch"]) for v in points if isinstance(v.get("epoch", None), int)]
    if len(epochs) == 0:
      return None
    metrics_to_show = [
      ("valid_loss", "#2563eb"),
      ("loss", "#f59e0b"),
      ("acc", "#10b981"),
      ("auc", "#ef4444"),
      ("kappa", "#8b5cf6"),
      ("f1", "#0ea5e9"),
    ]
    series = []
    for key, color in metrics_to_show:
      values = []
      for item in points:
        if not isinstance(item.get("epoch", None), int):
          continue
        if key == "valid_loss":
          value = item.get("valid_loss", None)
        else:
          value = (item.get("metrics", {}) or {}).get(key, None)
        values.append(float(value) if isinstance(value, (int, float)) else None)
      if any(v is not None for v in values):
        series.append(dict(name=key, color=color, y=values))
    if len(series) == 0:
      return None
    chart_id = self._next_chart_id("epoch_metric")
    return self._chart_svg_line(
      chart_id=chart_id,
      title="Epoch Metrics Trend",
      x_label="Epoch",
      y_label="Metric Value",
      x_values=[float(v) for v in epochs],
      series_list=series,
    )

  def _build_conclusion_section(
    self,
    *,
    epoch_records: List[Dict[str, Any]],
    best_valid_epoch: Optional[int],
    best_valid_loss: Optional[float],
    test_metrics: Dict[str, Any],
    test_result_file: Optional[Path],
  ):
    lines = []
    if best_valid_epoch is not None:
      loss_text = self._format_float(best_valid_loss) if isinstance(best_valid_loss, (int, float)) else "N/A"
      lines.append(f"最佳验证轮次为 Epoch {best_valid_epoch}，对应 valid loss 为 {loss_text}。")
    if len(test_metrics) > 0:
      auc_text = self._format_plain(test_metrics.get("auc", "N/A"))
      acc_text = self._format_plain(test_metrics.get("acc", "N/A"))
      f1_text = self._format_plain(test_metrics.get("f1", "N/A"))
      kappa_text = self._format_plain(test_metrics.get("kappa", "N/A"))
      lines.append(
        "测试集关键指标："
        f"AUC={auc_text}，ACC={acc_text}，F1={f1_text}，Kappa={kappa_text}。"
      )

    train_dist = (self.dataset_info.get("train") or {}).get("raw_label_distribution", {}) or {}
    if len(train_dist) > 1:
      counts = [int(v) for v in train_dist.values()]
      max_count = max(counts)
      min_count = max(1, min(counts))
      imbalance_ratio = max_count / min_count
      if imbalance_ratio >= 3.0:
        lines.append(f"训练集类别不均衡明显（最大/最小约 {self._format_float(imbalance_ratio)} 倍），建议加强重采样或类别加权。")

    if test_result_file is not None and Path(test_result_file).exists():
      labels, preds, _ = self._extract_eval_arrays(test_result_file)
      if labels is not None and preds is not None and len(labels) > 0:
        cm = confusion_matrix(labels, preds, labels=list(range(max(max(labels), max(preds)) + 1)))
        recalls = []
        for i in range(cm.shape[0]):
          denom = cm[i, :].sum()
          recall = 0.0 if denom <= 0 else float(cm[i, i]) / float(denom)
          recalls.append(recall)
        worst_index = int(np.argmin(recalls)) if len(recalls) > 0 else 0
        if len(recalls) > 0:
          worst_label = self.display_class_names[worst_index] if worst_index < len(self.display_class_names) else str(worst_index)
          lines.append(
            f"分类薄弱类别为 {worst_label}（召回约 {self._format_float(100.0 * recalls[worst_index])}%），"
            "建议增加该类样本、进行针对性增强并在损失函数中提高该类权重。"
          )

    lines.append("后续优化方向：可尝试 focal loss + class weight、分层抽样、hard example mining，并结合阈值调优提升临床敏感类别召回。")
    items = "".join([f"<li>{line}</li>" for line in lines])
    return (
      "<div class='card' id='conclusion'>"
      "<h3>Conclusion And Next Steps</h3>"
      f"<ul class='insight-list'>{items}</ul>"
      "</div>"
    )

  def _render_header(self, *, title: str, subtitle: str):
    return (
      "<header class='hero'>"
      f"<h1>{html.escape(title)}</h1>"
      f"<p>{html.escape(subtitle)}</p>"
      "</header>"
      "<nav class='report-nav'>"
      "<a href='#summary'>Summary</a>"
      "<a href='#trend'>Trend</a>"
      "<a href='#figures'>Figures</a>"
      "<a href='#dataset'>Dataset</a>"
      "<a href='#conclusion'>Conclusion</a>"
      "</nav>"
    )

  def _save_html(self, path: Path, body: str):
    template = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Experiment Report</title>
  <style>
    :root {{
      --bg: #eef3fb;
      --card: #ffffff;
      --text: #0f172a;
      --muted: #475569;
      --line: #dbe5f3;
      --brand-a: #0b2f78;
      --brand-b: #1546b5;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif;
      line-height: 1.45;
    }}
    .container {{ max-width: 1280px; margin: 0 auto; padding: 20px; }}
    .hero {{
      border-radius: 14px;
      background: linear-gradient(120deg, var(--brand-a), var(--brand-b));
      color: #f8fafc;
      padding: 26px 28px;
      box-shadow: 0 10px 28px rgba(11, 47, 120, 0.28);
    }}
    .hero h1 {{ margin: 0; font-size: 34px; font-weight: 800; letter-spacing: 0.2px; }}
    .hero p {{ margin: 8px 0 0; color: rgba(248, 250, 252, 0.9); }}
    .report-nav {{
      margin-top: 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
    }}
    .report-nav a {{
      color: #1d4ed8;
      text-decoration: none;
      font-weight: 700;
      font-size: 13px;
      padding: 4px 8px;
      border-radius: 8px;
    }}
    .report-nav a:hover {{ background: #eff6ff; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      margin-top: 16px;
      box-shadow: 0 4px 18px rgba(15, 23, 42, 0.05);
    }}
    h3 {{ margin: 0 0 12px; font-size: 24px; }}
    h4 {{ margin: 18px 0 8px; font-size: 19px; }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
    }}
    .metric {{
      background: #f8fbff;
      border: 1px solid #d7e4fb;
      border-radius: 12px;
      padding: 10px 12px;
    }}
    .metric .label {{ font-size: 12px; color: var(--muted); font-weight: 600; }}
    .metric .value {{ margin-top: 2px; font-size: 22px; font-weight: 800; overflow-x: auto; white-space: nowrap; }}
    .line-chart-svg {{ width: 100%; height: auto; display: block; border-radius: 12px; background: #fff; border: 1px solid #dbe5f3; }}
    .chart-shell {{ border: 1px solid #d7e3f4; border-radius: 12px; padding: 10px; background: #fdfefe; }}
    .legend-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; }}
    .legend-item {{
      border: 1px solid #cdd8ea;
      background: #f8fafc;
      color: #0f172a;
      border-radius: 999px;
      padding: 5px 10px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 7px;
      font-size: 12px;
      font-weight: 700;
    }}
    .legend-item.active {{ background: #e9f1ff; border-color: #90b4f8; }}
    .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
    .grid-line {{ stroke: #e7edf8; stroke-width: 1; }}
    .axis-line {{ stroke: #94a3b8; stroke-width: 1.2; }}
    .axis-tick {{ fill: #475569; font-size: 11px; }}
    .axis-label {{ fill: #1f2937; font-size: 13px; font-weight: 600; }}
    .chart-title {{ fill: #0f172a; font-size: 18px; font-weight: 700; }}
    .scroll-x {{ overflow-x: auto; white-space: nowrap; width: 100%; }}
    pre {{ white-space: pre; background: #0f172a; color: #f8fafc; padding: 12px; border-radius: 10px; overflow: auto; }}
    table {{ border-collapse: collapse; width: 100%; }}
    .flat-table th, .flat-table td {{ border-bottom: 1px solid #e2e8f0; padding: 8px; text-align: left; vertical-align: top; }}
    .flat-table th {{ background: #f8fafc; min-width: 120px; }}
    .dataset-grid {{ display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 12px; margin-bottom: 12px; }}
    .split-card {{ border: 1px solid #d7e4fb; border-radius: 12px; padding: 10px; background: #fbfdff; }}
    .split-card h4 {{ margin: 0 0 8px; font-size: 17px; color: #1d4ed8; }}
    .split-card p {{ margin: 6px 0; display: grid; grid-template-columns: 88px 1fr; gap: 6px; }}
    .split-card .k {{ color: #64748b; font-weight: 700; }}
    .split-card .v {{ color: #0f172a; overflow: hidden; }}
    .compare-table th, .compare-table td {{ border-bottom: 1px solid #e2e8f0; padding: 8px; text-align: left; vertical-align: top; }}
    .compare-table th {{ background: #f8fafc; }}
    .ratio-line {{ font-size: 12px; color: #334155; }}
    .cm-wrap {{ border: 1px solid #d8e4f8; border-radius: 12px; padding: 10px; background: #fbfdff; }}
    .cm-title {{ font-size: 16px; font-weight: 800; margin-bottom: 8px; }}
    .cm-table th, .cm-table td {{ padding: 8px 10px; border-bottom: 1px solid #e2e8f0; text-align: center; white-space: nowrap; }}
    .cm-table th:first-child, .cm-table td:first-child {{ text-align: left; position: sticky; left: 0; background: #fff; z-index: 1; }}
    .insight-list {{ margin: 0; padding-left: 18px; display: grid; gap: 8px; }}
    @media (max-width: 980px) {{
      .dataset-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    {body}
  </div>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template, encoding="utf-8")
    return path

  def write_epoch_report(
    self,
    *,
    epoch: int,
    phase: str,
    metrics: Dict[str, Any],
    result_file: Path,
    checkpoint_file: Optional[Path] = None,
    best_valid_epoch: Optional[int] = None,
  ):
    metrics_plain = self._to_plain(metrics)
    figures = self.build_eval_figures(result_file=result_file, file_prefix=f"epoch{epoch}_{phase}")
    metric_cards = []
    for key, value in metrics_plain.items():
      metric_cards.append(
        f"<div class='metric'><div class='label'>{html.escape(str(key))}</div>"
        f"<div class='value'>{self._render_value_html(value)}</div></div>"
      )

    eval_blocks = []
    for key in ["roc", "pr", "cm"]:
      if key in figures:
        eval_blocks.append(f"<h4>{key.upper()}</h4>{figures[key]}")
    header = self._render_header(
      title=f"Epoch {epoch} Report",
      subtitle=f"phase={phase} | best_valid_epoch={best_valid_epoch}",
    )
    body = (
      f"{header}"
      "<section class='card' id='summary'>"
      "<h3>Summary</h3>"
      f"<div class='metric-grid'>{''.join(metric_cards)}</div>"
      "</section>"
      "<section class='card'>"
      "<h3>Artifacts</h3>"
      f"{self._render_dict_as_table(dict(result_file=str(result_file), checkpoint_file=str(checkpoint_file) if checkpoint_file else 'N/A', best_valid_epoch=best_valid_epoch))}"
      "</section>"
      f"<section class='card' id='figures'><h3>Eval Figures</h3>{''.join(eval_blocks) if len(eval_blocks) > 0 else '<p>No figures generated.</p>'}</section>"
      f"<section class='card' id='dataset'><h3>Dataset Distribution</h3>{self._render_dataset_section()}</section>"
      f"<section class='card'><h3>Model</h3><pre>{html.escape(self.model_desc)}</pre></section>"
      f"<section class='card'><h3>MODEL Config</h3>{self._json_block(self.cfg.MODEL)}</section>"
      f"<section class='card'><h3>RUNNER Config</h3>{self._json_block(self.cfg.RUNNER)}</section>"
      f"<section class='card'><h3>METRIC Config</h3>{self._json_block(self.cfg.METRIC)}</section>"
    )
    report_file = self.results_dir / f"epoch{epoch}.html"
    self._save_html(report_file, body)
    return report_file

  def write_final_report(
    self,
    *,
    epoch_records: List[Dict[str, Any]],
    best_valid_epoch: Optional[int],
    best_valid_loss: Optional[float],
    best_valid_ckpt: Optional[Path],
    test_result_file: Optional[Path] = None,
    test_metrics: Optional[Dict[str, Any]] = None,
  ):
    test_metrics_plain = self._to_plain(test_metrics or {})
    valid_loss_chart = self._build_valid_loss_chart(epoch_records)
    epoch_metric_chart = self._build_epoch_metric_chart(epoch_records)
    test_figures = {}
    if test_result_file is not None and Path(test_result_file).exists():
      test_figures = self.build_eval_figures(result_file=test_result_file, file_prefix="test_best")

    summary_cards = []
    summary_items = {
      "best_valid_epoch": best_valid_epoch,
      "best_valid_loss": best_valid_loss,
      "best_ckpt": str(best_valid_ckpt) if best_valid_ckpt else "N/A",
      "test_loss": test_metrics_plain.get("loss", "N/A"),
      "test_acc": test_metrics_plain.get("acc", "N/A"),
      "test_auc": test_metrics_plain.get("auc", "N/A"),
      "test_kappa": test_metrics_plain.get("kappa", "N/A"),
      "test_f1": test_metrics_plain.get("f1", "N/A"),
    }
    for key, value in summary_items.items():
      summary_cards.append(
        f"<div class='metric'><div class='label'>{html.escape(str(key))}</div>"
        f"<div class='value'>{self._render_value_html(value)}</div></div>"
      )

    trend_blocks = []
    if valid_loss_chart is not None:
      trend_blocks.append("<h4>Valid Loss Trend</h4>" + valid_loss_chart)
    if epoch_metric_chart is not None:
      trend_blocks.append("<h4>Epoch Metrics Trend</h4>" + epoch_metric_chart)

    eval_blocks = []
    for key in ["roc", "pr", "cm"]:
      if key in test_figures:
        eval_blocks.append(f"<h4>{key.upper()}</h4>{test_figures[key]}")

    header = self._render_header(
      title="BREXI Experiment Report",
      subtitle=f"best_epoch={best_valid_epoch} | generated_by=slimai.runner.report",
    )
    body = (
      f"{header}"
      "<section class='card' id='summary'>"
      "<h3>Summary</h3>"
      f"<div class='metric-grid'>{''.join(summary_cards)}</div>"
      "</section>"
      f"<section class='card' id='trend'><h3>Trend Analysis</h3>{''.join(trend_blocks) if len(trend_blocks) > 0 else '<p>N/A</p>'}</section>"
      f"<section class='card'><h3>Epoch Metrics Table</h3>{self._render_epoch_table(epoch_records)}</section>"
      f"<section class='card' id='figures'><h3>Test Eval Figures</h3>{''.join(eval_blocks) if len(eval_blocks) > 0 else '<p>No figures generated.</p>'}</section>"
      f"<section class='card'><h3>Test Metrics</h3>{self._render_dict_as_table(test_metrics_plain)}</section>"
      f"<section class='card' id='dataset'><h3>Dataset Distribution</h3>{self._render_dataset_section()}</section>"
      f"{self._build_conclusion_section(epoch_records=epoch_records, best_valid_epoch=best_valid_epoch, best_valid_loss=best_valid_loss, test_metrics=test_metrics_plain, test_result_file=test_result_file)}"
    )
    report_file = self.work_dir / "report.html"
    self._save_html(report_file, body)
    print_log(f"Saved final report to: {report_file}")
    return report_file
