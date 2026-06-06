from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Optional


def _esc(value: Any) -> str:
  return html.escape("" if value is None else str(value))


def _fmt_float(value: Any, digits: int = 6) -> str:
  if value is None:
    return "N/A"
  if isinstance(value, str):
    return value
  try:
    return f"{float(value):.{digits}f}"
  except Exception:
    return str(value)


def _fmt_tol(value: Any) -> str:
  if value is None:
    return "N/A"
  try:
    v = float(value)
    if v == 0.0:
      return "0"
    if abs(v) >= 1e-3:
      return f"{v:g}"
    return f"{v:.0e}"
  except Exception:
    return str(value)


def _render_threshold_line(text: str) -> str:
  return f"<p class='threshold'><strong>阈值</strong>：{text}</p>"


def _badge(passed: Any) -> str:
  ok = bool(passed)
  cls = "pass" if ok else "fail"
  text = "PASS" if ok else "FAIL"
  return f"<span class='badge {cls}'>{text}</span>"


def _render_table(rows: List[List[Any]], *, headers: List[str], row_header: bool = False) -> str:
  head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
  body_rows = []
  for row in rows:
    cells = []
    for idx, cell in enumerate(row):
      if row_header and idx == 0:
        cells.append(f"<th>{_esc(cell)}</th>")
      else:
        cells.append(f"<td>{_esc(cell)}</td>")
    body_rows.append(f"<tr>{''.join(cells)}</tr>")
  return f"<table class='flat-table compare-table'><tr>{head}</tr>{''.join(body_rows)}</table>"


def _render_dict_table(data: Dict[str, Any]) -> str:
  if not data:
    return "<p>N/A</p>"
  rows = [[k, _fmt_float(v) if isinstance(v, (int, float)) else v] for k, v in data.items()]
  return _render_table(rows, headers=["metric", "value"], row_header=True)


def _render_reference_compare(rows: List[Dict[str, Any]], *, metrics_tol: Any = None) -> str:
  if not rows:
    return "<p>No baseline comparison.</p>"
  parts = []
  if metrics_tol is not None:
    parts.append(_render_threshold_line(f"|ONNX − Original| < {_fmt_tol(metrics_tol)} 视为 match"))
  table_rows = []
  for row in rows:
    match = row.get("match")
    table_rows.append([
      row.get("metric"),
      _fmt_float(row.get("onnx")),
      _fmt_float(row.get("original")),
      _fmt_float(row.get("delta")),
      _fmt_tol(metrics_tol) if metrics_tol is not None else "N/A",
      "yes" if match else "no",
    ])
  parts.append(_render_table(
    table_rows,
    headers=["metric", "ONNX (V4)", "Original Report", "delta", "tol", "match"],
  ))
  return "".join(parts)


def _resolve_parity_tols(summary: Dict[str, Any]) -> tuple[Any, Any]:
  parity_max_tol = summary.get("parity_max_tol", summary.get("parity_tol", 5e-5))
  parity_mean_tol = summary.get("parity_mean_tol", 5e-6)
  return parity_max_tol, parity_mean_tol


def _render_parity_output_block(
  output_name: str,
  block: Dict[str, Any],
  *,
  parity_max_tol: Any = None,
  parity_mean_tol: Any = None,
) -> str:
  parts = [f"<h5>{_esc(output_name)}</h5>"]
  if block.get("kind") == "int":
    parts.append(_render_threshold_line("整数输出：PT 与 ORT 完全一致（exact_match）"))
    trial_rows = []
    for trial in block.get("trials", []):
      trial_rows.append([
        trial.get("batch_size"),
        "yes" if trial.get("exact_match") else "no",
      ])
    parts.append(_render_table(trial_rows, headers=["batch_size N", "exact_match"]))
    return "".join(parts)
  max_tol_text = _fmt_tol(parity_max_tol) if parity_max_tol is not None else "N/A"
  mean_tol_text = _fmt_tol(parity_mean_tol) if parity_mean_tol is not None else "N/A"
  parts.append(
    _render_threshold_line(
      f"浮点输出：max |PT − ORT| < {max_tol_text} 且 mean |PT − ORT| < {mean_tol_text}"
    )
  )
  trial_rows = []
  for trial in block.get("trials", []):
    err = trial.get("error", {})
    trial_rows.append([
      trial.get("batch_size"),
      _fmt_float(err.get("min"), 8),
      _fmt_float(err.get("max"), 8),
      _fmt_float(err.get("mean"), 8),
      _fmt_float(err.get("variance"), 8),
      max_tol_text,
      mean_tol_text,
    ])
  parts.append(
    "<h6>Per-trial error (|pt-ort|)</h6>"
    + _render_table(
      trial_rows,
      headers=["N", "min", "max", "mean", "variance", "max_tol", "mean_tol"],
    )
  )
  agg = block.get("aggregate", {})
  parts.append(
    "<h6>Cross-trial aggregate (on trial max)</h6>"
    + _render_table(
      [[
        _fmt_float(agg.get("min"), 8),
        _fmt_float(agg.get("max"), 8),
        _fmt_float(agg.get("mean"), 8),
        _fmt_float(agg.get("variance"), 8),
        max_tol_text,
      ]],
      headers=["min", "max", "mean", "variance", "max_tol"],
    )
  )
  return "".join(parts)


def _render_v1_hf_section(
  block: Dict[str, Any],
  *,
  parity_max_tol: Any = None,
  parity_mean_tol: Any = None,
  weight_max_tol: Any = None,
) -> str:
  weight = block.get("weight", {})
  w_tol = _fmt_tol(weight_max_tol if weight_max_tol is not None else weight.get("tol"))
  max_tol_text = _fmt_tol(parity_max_tol) if parity_max_tol is not None else "N/A"
  mean_tol_text = _fmt_tol(parity_mean_tol) if parity_mean_tol is not None else "N/A"
  parts = [
    f"<section class='card'><h3>V1-HF 预训练 UNI vs patch_encoder (PyTorch) "
    f"{_badge(block.get('passed'))}</h3>",
    _render_threshold_line(
      f"权重 max |HF.encoder − patch_encoder.encoder| < {w_tol}；"
      f"随机 patch 前向 max |HF − PT| < {max_tol_text} 且 mean < {mean_tol_text}"
    ),
    f"<p>encoder={_esc(block.get('encoder_name'))}; "
    f"shared_keys={_esc(weight.get('shared_keys'))}</p>",
    f"<p>权重 max_diff={_fmt_float(weight.get('max_diff'), 8)} "
    f"(worst_key={_esc(weight.get('worst_key'))}, tol={w_tol}) "
    f"{_badge(weight.get('passed'))}</p>",
    f"<p>num_trials={_esc(block.get('num_trials'))}; "
    f"batch_size_range={_esc(block.get('batch_size_range'))}; "
    f"per_trial_batch_size={_esc(block.get('per_trial_batch_size'))}</p>",
  ]
  for output_name, output_block in (block.get("outputs") or {}).items():
    parts.append(
      _render_parity_output_block(
        str(output_name),
        output_block,
        parity_max_tol=parity_max_tol,
        parity_mean_tol=parity_mean_tol,
      )
    )
  parts.append("</section>")
  return "".join(parts)


def _render_parity_section(
  title: str,
  block: Dict[str, Any],
  *,
  parity_max_tol: Any = None,
  parity_mean_tol: Any = None,
) -> str:
  max_tol_text = _fmt_tol(parity_max_tol) if parity_max_tol is not None else "N/A"
  mean_tol_text = _fmt_tol(parity_mean_tol) if parity_mean_tol is not None else "N/A"
  parts = [
    f"<section class='card'><h3>{_esc(title)} {_badge(block.get('passed'))}</h3>",
    _render_threshold_line(
      f"浮点 max |PT − ORT| < {max_tol_text} 且 mean |PT − ORT| < {mean_tol_text}；整数标签 exact_match"
    ),
    f"<p>num_trials={_esc(block.get('num_trials'))}; "
    f"batch_size_range={_esc(block.get('batch_size_range'))}; "
    f"per_trial_batch_size={_esc(block.get('per_trial_batch_size'))}</p>",
  ]
  if block.get("calibration_files"):
    parts.append(f"<p>Calibration: {_esc(', '.join(block['calibration_files']))}</p>")
  for output_name, output_block in (block.get("outputs") or {}).items():
    parts.append(
      _render_parity_output_block(
        str(output_name),
        output_block,
        parity_max_tol=parity_max_tol,
        parity_mean_tol=parity_mean_tol,
      )
    )
  parts.append("</section>")
  return "".join(parts)


def _render_l0_section(block: Dict[str, Any]) -> str:
  rows = []
  for model_key in ["patch_encoder", "slide_encoder"]:
    item = block.get(model_key, {})
    rows.append([
      model_key,
      item.get("checker_passed"),
      item.get("simplify_applied"),
      item.get("checker_error") or "",
    ])
  return (
    f"<section class='card'><h3>L0 ONNX Check {_badge(block.get('passed'))}</h3>"
    + _render_threshold_line("ONNX checker 结构/版本检查通过（无数值容差）")
    + _render_table(rows, headers=["model", "checker_passed", "simplify_applied", "checker_error"])
    + "</section>"
  )


def _render_l1_section(block: Dict[str, Any], *, deterministic_tol: Any = None) -> str:
  patch = block.get("patch_encoder_onnx_deterministic", {})
  slide = block.get("slide_encoder_onnx_deterministic", {})
  tol_text = _fmt_tol(deterministic_tol) if deterministic_tol is not None else "N/A"
  parts = [
    f"<section class='card'><h3>L1 ORT Deterministic {_badge(block.get('passed'))}</h3>",
    _render_threshold_line(f"同一输入重复 ORT 推理：max_diff < {tol_text}"),
    f"<p>repeats={_esc(patch.get('repeats'))}; backend={_esc(patch.get('backend'))}</p>",
    f"<p>patch_encoder max_diff={_fmt_float(patch.get('max_diff'), 8)} "
    f"(tol={tol_text}) {_badge(patch.get('passed'))}</p>",
  ]
  slide_diffs = slide.get("max_diff_by_output", {})
  if slide_diffs:
    diff_rows = [[k, _fmt_float(v, 8), tol_text] for k, v in slide_diffs.items()]
    parts.append(_render_table(diff_rows, headers=["output", "max_diff", "tol"], row_header=True))
  parts.append("</section>")
  return "".join(parts)


def _render_figures(figures: Dict[str, str]) -> str:
  if not figures:
    return "<p>No figures.</p>"
  chunks = []
  for key, fig_html in figures.items():
    chunks.append(f"<div class='figure-block'><h6>{_esc(key.upper())}</h6>{fig_html}</div>")
  return "".join(chunks)


def _render_label_agreement(agreement: Dict[str, Any], *, label_agreement_tol: float = 1.0) -> str:
  rates = agreement.get("rates", {})
  if not rates:
    return "<p>N/A</p>"
  tol_text = _fmt_tol(label_agreement_tol)
  rows = [
    [k, _fmt_float(v, 4), tol_text, "yes" if float(v) >= label_agreement_tol else "no"]
    for k, v in rates.items()
  ]
  return (
    _render_threshold_line(f"PyTorch slide vs ORT（同 embedding）：match_rate ≥ {tol_text}")
    + f"<p>Label agreement: {_badge(agreement.get('passed'))}</p>"
    + _render_table(rows, headers=["label_type", "match_rate", "tol", "passed"], row_header=True)
  )


def _render_metric_parity_table(parity_block: Dict[str, Any]) -> str:
  metrics = parity_block.get("metrics", {})
  if not metrics:
    return "<p>N/A</p>"
  rows = []
  for key, item in sorted(metrics.items()):
    rows.append([
      key,
      _fmt_float(item.get("pt")),
      _fmt_float(item.get("ort")),
      _fmt_float(item.get("delta")),
      _fmt_tol(item.get("tol")),
      "yes" if item.get("match") else "no",
    ])
  return _render_table(
    rows,
    headers=["metric", "PT", "ORT", "delta", "tol", "match"],
    row_header=True,
  )


def _render_v4_task_block(task_name: str, task_block: Dict[str, Any]) -> str:
  parts = [
    f"<div class='v4-task'><h5>{_esc(task_name)} {_badge(task_block.get('passed'))}</h5>",
    "<h6>Metric Parity (PT vs ORT)</h6>",
    _render_metric_parity_table(task_block.get("metric_parity", {})),
  ]
  figures = task_block.get("figures_ort", {})
  if figures:
    parts.append("<h6>ORT Figures</h6>" + _render_figures(figures))
  parts.append("</div>")
  return "".join(parts)


def _render_v4_subset(subset_name: str, block: Dict[str, Any], *, metrics_tol: Any = None) -> str:
  parts = [
    f"<div class='v4-subset'><h4>{_esc(subset_name)} {_badge(block.get('passed'))}</h4>",
    f"<p>n_samples={_esc(block.get('n_samples'))}</p>",
  ]
  if block.get("note"):
    parts.append(f"<p class='muted'>{_esc(block.get('note'))}</p>")
  if block.get("primary"):
    parts.append(_render_v4_task_block("primary", block["primary"]))
  if block.get("marginal"):
    parts.append(_render_v4_task_block("marginal", block["marginal"]))
  if block.get("conditional"):
    parts.append(_render_v4_task_block("conditional", block["conditional"]))
  if block.get("label_agreement"):
    parts.append("<h5>Label agreement</h5>" + _render_label_agreement(block["label_agreement"]))
  by_center = block.get("by_center", {})
  if by_center:
    parts.append("<h5>Inner test by center</h5>")
    for center_name, center_block in sorted(by_center.items()):
      parts.append(f"<div class='by-center'><h6>{_esc(center_name)}</h6>")
      parts.append(_render_dict_table(center_block.get("metrics", {})))
      if center_block.get("label_agreement"):
        parts.append(_render_label_agreement(center_block["label_agreement"]))
      if center_block.get("figures"):
        parts.append(_render_figures(center_block["figures"]))
      parts.append("</div>")
  parts.append("</div>")
  return "".join(parts)


def _render_v4_section(block: Dict[str, Any], *, metrics_tol: Any = None) -> str:
  if block.get("skipped"):
    return (
      f"<section class='card'><h3>V4 Test Evaluation</h3>"
      f"<p>Skipped: {_esc(block.get('reason', 'unknown'))}</p></section>"
    )
  baseline = block.get("baseline") or {}
  warnings = baseline.get("warnings", [])
  tol_text = _fmt_tol(metrics_tol) if metrics_tol is not None else "N/A"
  parts = [
    f"<section class='card'><h3>V4 Test Evaluation {_badge(block.get('passed'))}</h3>",
    _render_threshold_line(
      f"数据集级 metric parity：|PT − ORT| < {tol_text}（primary / marginal / conditional）；"
      "PT/ORT label agreement：match_rate ≥ 1.0"
    ),
    f"<p>external_count={_esc(block.get('external_count', 0))}</p>",
  ]
  if warnings:
    parts.append(f"<p class='warn'>Baseline warnings: {_esc('; '.join(warnings))}</p>")
  if baseline.get("work_dir"):
    parts.append(f"<p>Baseline work_dir: {_esc(baseline['work_dir'])}</p>")
  for subset_name, subset_block in (block.get("subsets") or {}).items():
    parts.append(_render_v4_subset(str(subset_name), subset_block, metrics_tol=metrics_tol))
  parts.append("</section>")
  return "".join(parts)


def render_validation_report_html(report_data: Dict[str, Any]) -> str:
  summary = report_data.get("summary", {})
  parity_max_tol, parity_mean_tol = _resolve_parity_tols(summary)
  deterministic_tol = summary.get("deterministic_tol")
  metrics_tol = summary.get("metrics_tol", 5e-4)
  timing = report_data.get("timing", {}).get("phases", {})
  timing_rows = [[k, timing[k]] for k in sorted(timing.keys())]

  sections = [
    "<section class='card'><h3>Summary</h3>",
    f"<p>policy: {_esc(summary.get('policy'))}; overall {_badge(summary.get('passed'))}</p>",
    _render_threshold_line(
      f"L1 deterministic_tol={_fmt_tol(deterministic_tol)}；"
      f"V1–V3 max_tol={_fmt_tol(parity_max_tol)}、mean_tol={_fmt_tol(parity_mean_tol)}；"
      f"V4 metrics_tol={_fmt_tol(metrics_tol)}；"
      "V4 label agreement ≥ 1.0"
    ),
    f"<p>num_trials={_esc(summary.get('num_trials'))}; ort_provider={_esc(summary.get('ort_provider'))}</p>",
  ]
  if summary.get("export"):
    sections.append("<h4>Export</h4>" + _render_dict_table(summary["export"]))
  if timing_rows:
    sections.append("<h4>Phase timing (sec)</h4>" + _render_table(timing_rows, headers=["phase", "elapsed_sec"]))
  sections.append("</section>")

  if report_data.get("l0"):
    sections.append(_render_l0_section(report_data["l0"]))
  if report_data.get("l1"):
    sections.append(_render_l1_section(report_data["l1"], deterministic_tol=deterministic_tol))
  if report_data.get("v1_hf"):
    from slimai.export.validate_ort import HF_WEIGHT_MAX_TOL

    sections.append(
      _render_v1_hf_section(
        report_data["v1_hf"],
        parity_max_tol=parity_max_tol,
        parity_mean_tol=parity_mean_tol,
        weight_max_tol=HF_WEIGHT_MAX_TOL,
      )
    )
  for key, title in [("v1", "V1 patch_encoder ORT parity"), ("v2", "V2 slide_encoder ORT parity"), ("v3", "V3 e2e ORT parity")]:
    if report_data.get(key):
      sections.append(
        _render_parity_section(
          title,
          report_data[key],
          parity_max_tol=parity_max_tol,
          parity_mean_tol=parity_mean_tol,
        )
      )
  if report_data.get("v4") is not None:
    sections.append(_render_v4_section(report_data["v4"], metrics_tol=metrics_tol))

  css = """
  body { font-family: Inter, Arial, sans-serif; margin: 20px; color: #0f172a; background: #eef3fb; }
  .card { background: #fff; border: 1px solid #dbe5f3; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
  .flat-table { border-collapse: collapse; width: 100%; margin: 8px 0 16px; }
  .flat-table th, .flat-table td { border: 1px solid #dbe5f3; padding: 6px 8px; text-align: left; font-size: 13px; }
  .compare-table th { background: #f8fbff; }
  .badge { font-size: 12px; padding: 2px 8px; border-radius: 8px; font-weight: 700; }
  .badge.pass { background: #dcfce7; color: #166534; }
  .badge.fail { background: #fee2e2; color: #991b1b; }
  .threshold { background: #f0f9ff; border-left: 3px solid #0284c7; padding: 8px 12px; margin: 8px 0 12px; font-size: 13px; color: #0c4a6e; }
  .muted { color: #64748b; }
  .warn { color: #b45309; }
  .v4-subset { border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; margin: 12px 0; }
  .v4-task { margin: 8px 0 12px; padding: 8px 0 8px 12px; border-left: 3px solid #dbe5f3; }
  .by-center { margin: 8px 0 12px 16px; padding-left: 12px; border-left: 3px solid #dbe5f3; }
  .figure-block { margin: 8px 0 16px; }
  .cm-wrap { overflow-x: auto; }
  h1 { margin-bottom: 16px; }
  """
  body = "".join(sections)
  return (
    f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
    f"<title>ONNX Export Validation Report</title><style>{css}</style></head>"
    f"<body><h1>ONNX Export Validation Report</h1>{body}</body></html>"
  )


def write_validation_report_html(output_dir: Path, report_data: Dict[str, Any]) -> Path:
  output_dir = Path(output_dir)
  html_content = render_validation_report_html(report_data)
  path = output_dir / "validation_main.html"
  path.write_text(html_content, encoding="utf-8")
  return path
