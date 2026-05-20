import hashlib
import json
import math
import random
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import mmengine
import pandas as pd

try:
  from slimai.helper.help_utils import print_log as _print_log
except Exception:
  _print_log = None


def _log(message: str):
  if callable(_print_log):
    _print_log(message)
    return
  print(message)
  return


def _to_plain(value: Any):
  try:
    import torch  # local import to keep module import lightweight

    if isinstance(value, torch.Tensor):
      value = value.detach().cpu()
      if value.numel() == 1:
        return float(value.item())
      return value.tolist()
  except Exception:
    pass

  try:
    import numpy as np  # local import for optional numpy scalars

    if isinstance(value, np.ndarray):
      return value.tolist()
    if isinstance(value, np.generic):
      return value.item()
  except Exception:
    pass

  if isinstance(value, (list, tuple)):
    return [_to_plain(v) for v in value]
  if isinstance(value, dict):
    return {str(k): _to_plain(v) for k, v in value.items()}
  return value


def _as_dict(item: Any) -> Dict[str, Any]:
  if isinstance(item, dict):
    return item
  if hasattr(item, "to_dict"):
    try:
      data = item.to_dict()
      if isinstance(data, dict):
        return data
    except Exception:
      pass
  return {}


def _as_int(value: Any) -> Optional[int]:
  value = _to_plain(value)
  if isinstance(value, bool):
    return int(value)
  if isinstance(value, int):
    return value
  if isinstance(value, float):
    if math.isnan(value):
      return None
    return int(value)
  return None


def _as_float(value: Any) -> Optional[float]:
  value = _to_plain(value)
  if isinstance(value, bool):
    return float(int(value))
  if isinstance(value, (int, float)):
    if isinstance(value, float) and math.isnan(value):
      return None
    return float(value)
  return None


def _as_bool(value: Any) -> bool:
  if isinstance(value, bool):
    return value
  if isinstance(value, (int, float)):
    return int(value) != 0
  if isinstance(value, str):
    text = value.strip().lower()
    return text in {"1", "true", "yes", "y", "t"}
  return False


def _to_list(value: Any) -> List[Any]:
  value = _to_plain(value)
  if value is None:
    return []
  if isinstance(value, list):
    return value
  if isinstance(value, tuple):
    return list(value)
  return [value]


def _pick_dict(item: Dict[str, Any], key: str) -> Dict[str, Any]:
  value = item.get(key, {})
  if isinstance(value, dict):
    return value
  return {}


def _get_label_name(label: Optional[int], class_names: List[str]) -> str:
  if label is None or label < 0:
    return ""
  if label < len(class_names):
    return str(class_names[label])
  return str(label)


def _resolve_h5_path(sample: Dict[str, Any]) -> str:
  meta = _pick_dict(sample, "meta")
  for key in ["h5_path", "file", "path"]:
    value = meta.get(key, None)
    if isinstance(value, str) and value.strip():
      return value.strip()
  for key in ["file", "path"]:
    value = sample.get(key, None)
    if isinstance(value, str) and value.strip():
      return value.strip()
  return ""


def _normalize_output(sample: Dict[str, Any], task_key: str) -> Dict[str, Any]:
  output = sample.get("output", {})
  if not isinstance(output, dict):
    return {}
  if task_key in output and isinstance(output.get(task_key), dict):
    return _pick_dict(output, task_key)
  return output


def _extract_attention_vector(output_root: Dict[str, Any], meta: Dict[str, Any]) -> List[Optional[float]]:
  for key in ["attentions", "attention", "patch_attentions", "atten_weights"]:
    if key in output_root:
      values = _to_list(output_root.get(key))
      out = []
      for value in values:
        out.append(_as_float(value))
      return out
  for key in ["attentions", "patch_attentions"]:
    if key in meta:
      values = _to_list(meta.get(key))
      out = []
      for value in values:
        out.append(_as_float(value))
      return out
  return []


def _extract_topk(output_root: Dict[str, Any], key_indices: str, key_scores: str) -> List[Dict[str, Any]]:
  indices = _to_list(output_root.get(key_indices, []))
  scores = _to_list(output_root.get(key_scores, []))
  if len(indices) == 1 and isinstance(indices[0], list):
    indices = indices[0]
  if len(scores) == 1 and isinstance(scores[0], list):
    scores = scores[0]
  size = max(len(indices), len(scores))
  records: List[Dict[str, Any]] = []
  for i in range(size):
    patch_id = _as_int(indices[i]) if i < len(indices) else None
    score = _as_float(scores[i]) if i < len(scores) else None
    records.append(
      {
        "rank": i,
        "patch_id": patch_id,
        "attention_score": score,
      }
    )
  return records


def _build_patches(coords: List[Any], attentions: List[Optional[float]]) -> List[Dict[str, Any]]:
  patches: List[Dict[str, Any]] = []
  for idx, coord in enumerate(coords):
    coord_list = _to_list(coord)
    patch = {
      "patch_id": idx,
      "coords": [(_as_float(v) if _as_float(v) is not None else v) for v in coord_list],
      "attention": attentions[idx] if idx < len(attentions) else None,
    }
    patches.append(patch)
  return patches


def _as_rel_if_possible(path_str: str, *, base_dir: Path) -> str:
  value = str(path_str or "").strip()
  if value == "":
    return ""
  p = Path(value)
  try:
    if p.is_absolute():
      return str(p.resolve().relative_to(base_dir.resolve()))
  except Exception:
    return value
  return value


def _detail_filename(split: str, row: Dict[str, Any]) -> str:
  indice = row.get("indice", None)
  gt_label = row.get("gt_label", None)
  pred_label = row.get("pred_label", None)
  h5_path = str(row.get("h5_path", ""))
  gt_tag = f"gt{gt_label}" if isinstance(gt_label, int) and gt_label >= 0 else "gtNA"
  pred_tag = f"pred{pred_label}" if isinstance(pred_label, int) and pred_label >= 0 else "predNA"
  indice_tag = f"{int(indice):06d}" if isinstance(indice, int) else "NA"
  digest = hashlib.sha1(
    f"{split}|{indice_tag}|{h5_path}|{gt_tag}|{pred_tag}".encode("utf-8")
  ).hexdigest()[:12]
  return f"{split}_{gt_tag}_{pred_tag}_{indice_tag}_{digest}.json"


def _find_existing_detail_json(
  detail_dir: Path,
  work_dir: Path,
  split_name: str,
  row: Dict[str, Any],
) -> Optional[Tuple[Path, str]]:
  """Reuse stable-name detail JSON to avoid duplicate writes."""
  stable = detail_dir / _detail_filename(split_name, row)
  try:
    if stable.exists() and stable.is_file() and stable.stat().st_size > 0:
      return stable, str(stable.relative_to(work_dir))
  except Exception:
    pass
  return None


def _json_detail_linked_counts_from_manifest_df(df: pd.DataFrame) -> Dict[str, int]:
  """Count rows with non-empty detail_json per split (final manifest sheet view)."""
  if not isinstance(df, pd.DataFrame) or len(df) == 0:
    return {}
  if "split" not in df.columns or "detail_json" not in df.columns:
    return {}
  tmp = df.copy()
  tmp["_split_key"] = tmp["split"].astype(str)
  tmp["_has_detail"] = tmp["detail_json"].notna() & tmp["detail_json"].astype(str).str.strip().ne("")
  grouped = tmp.groupby("_split_key", dropna=False)["_has_detail"].sum()
  return {str(k): int(v) for k, v in grouped.items()}


def _progress(iterable, *, total: Optional[int], desc: str, enabled: bool, unit: str):
  if not enabled:
    return iterable
  try:
    from tqdm import tqdm

    return tqdm(iterable, total=total, desc=desc, unit=unit, leave=False)
  except Exception:
    return iterable


def _write_json_text(path: Path, text: str):
  path.write_text(text, encoding="utf-8")
  return


def _write_json_obj(path: Path, payload: Dict[str, Any], *, compact: bool = True):
  if compact:
    path.write_text(
      json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
      encoding="utf-8",
    )
  else:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
  return


def _normalize_export_detail_for(value: str) -> str:
  mode = str(value or "sampled").strip().lower()
  if mode not in {"all", "sampled", "none"}:
    raise ValueError(f"export_detail_for must be all|sampled|none, got: {value}")
  return mode


def _detail_indices_for_export(
  *,
  export_detail_for: str,
  num_rows: int,
  selected_reasons: Dict[int, Set[str]],
) -> List[int]:
  mode = _normalize_export_detail_for(export_detail_for)
  if mode == "none":
    return []
  if mode == "all":
    return list(range(num_rows))
  return sorted(selected_reasons.keys())


def _reasons_for_detail_index(
  *,
  export_detail_for: str,
  idx: int,
  selected_reasons: Dict[int, Set[str]],
) -> List[str]:
  mode = _normalize_export_detail_for(export_detail_for)
  reasons = sorted(selected_reasons.get(idx, set()))
  if mode == "all" and "all_attention" not in reasons:
    reasons.append("all_attention")
  return reasons


def load_batch_items(result_path: Path) -> List[Dict[str, Any]]:
  payload = mmengine.load(Path(result_path))
  if not isinstance(payload, dict):
    return []
  items = payload.get("batch_info", [])
  if not isinstance(items, list):
    return []
  out: List[Dict[str, Any]] = []
  for item in items:
    data = _as_dict(item)
    if isinstance(data, dict) and len(data) > 0:
      out.append(data)
  return out


def _build_row_summary(
  item: Dict[str, Any],
  *,
  split: str,
  class_names: List[str],
  secondary_class_names: List[str],
  source_pkl: Path,
  dataset_split: str = "",
  dataset_source: str = "",
  analysis_result_file: str = "",
) -> Dict[str, Any]:
  sample = _as_dict(item)
  label_output = _normalize_output(sample, "label")
  secondary_output = _normalize_output(sample, "label_secondary")
  secondary_conditional_output = _normalize_output(sample, "label_secondary_conditional")

  gt_label = _as_int(sample.get("label"))
  pred_label = _as_int(label_output.get("labels"))
  score = _as_float(label_output.get("scores"))

  gt_label_secondary = _as_int(sample.get("label_secondary"))
  pred_label_secondary = _as_int(secondary_output.get("labels"))
  score_secondary = _as_float(secondary_output.get("scores"))
  pred_label_secondary_conditional = _as_int(secondary_conditional_output.get("labels"))

  row = {
    "h5_path": _resolve_h5_path(sample),
    "split": split,
    "dataset_split": dataset_split or split,
    "dataset_source": dataset_source,
    "analysis_result_file": analysis_result_file,
    "indice": _as_int(sample.get("indice")),
    "source_pkl": str(source_pkl),
    "gt_label": gt_label,
    "gt_label_name": _get_label_name(gt_label, class_names),
    "pred_label": pred_label,
    "pred_label_name": _get_label_name(pred_label, class_names),
    "score": score,
    "gt_label_secondary": gt_label_secondary,
    "gt_label_secondary_name": _get_label_name(gt_label_secondary, secondary_class_names),
    "pred_label_secondary": pred_label_secondary,
    "pred_label_secondary_name": _get_label_name(pred_label_secondary, secondary_class_names),
    "score_secondary": score_secondary,
    "pred_label_secondary_conditional": pred_label_secondary_conditional,
    "pred_label_secondary_conditional_name": _get_label_name(pred_label_secondary_conditional, secondary_class_names),
    "is_mismatch": int(
      gt_label is not None
      and pred_label is not None
      and gt_label >= 0
      and pred_label >= 0
      and gt_label != pred_label
    ),
    "confusion_gt_label": gt_label,
    "confusion_pred_label": pred_label,
    "confusion_cell_key": f"{gt_label}|{pred_label}",
    "prediction_score_for_sampling": score,
    "detail_json_generated": 0,
    "detail_json_sample_reason": "",
    "detail_json": "",
  }
  return row


def _build_detail_record(
  item: Dict[str, Any],
  *,
  row: Dict[str, Any],
  split: str,
  class_names: List[str],
  secondary_class_names: List[str],
  source_pkl: Path,
  reasons: List[str],
  sample_random_seed: int,
  include_raw_output: bool = False,
  include_raw_meta: bool = False,
) -> Dict[str, Any]:
  sample = _as_dict(item)
  meta = _pick_dict(sample, "meta")
  output_root = _pick_dict(sample, "output")

  gt_label = row.get("gt_label", None)
  pred_label = row.get("pred_label", None)
  score = row.get("score", None)
  gt_label_secondary = row.get("gt_label_secondary", None)
  pred_label_secondary = row.get("pred_label_secondary", None)
  pred_label_secondary_conditional = row.get("pred_label_secondary_conditional", None)
  score_secondary = row.get("score_secondary", None)
  h5_path = row.get("h5_path", "")

  coords = _to_list(meta.get("coords", []))
  attentions = _extract_attention_vector(output_root, meta)

  detail = {
    "schema_version": "sample_analysis_detail_v2",
    "split": split,
    "indice": row.get("indice", None),
    "h5_path": h5_path,
    "sampling": {
      "confusion_gt_label": gt_label,
      "confusion_pred_label": pred_label,
      "prediction_score": score,
      "reasons": list(reasons),
      "seed": int(sample_random_seed),
    },
    "label": {
      "gt": gt_label,
      "gt_name": _get_label_name(gt_label, class_names),
      "pred": pred_label,
      "pred_name": _get_label_name(pred_label, class_names),
      "score": score,
    },
    "label_secondary": {
      "gt": gt_label_secondary,
      "gt_name": _get_label_name(gt_label_secondary, secondary_class_names),
      "pred": pred_label_secondary,
      "pred_name": _get_label_name(pred_label_secondary, secondary_class_names),
      "score": score_secondary,
      "conditional_pred": pred_label_secondary_conditional,
      "conditional_pred_name": _get_label_name(pred_label_secondary_conditional, secondary_class_names),
    },
    "topk": _extract_topk(output_root, "topk_atten_indices", "topk_atten_scores"),
    "tailk": _extract_topk(output_root, "tailk_atten_indices", "tailk_atten_scores"),
    "patches": _build_patches(coords, attentions),
    "meta": {
      "patch_num": _as_int(meta.get("patch_num")),
      "source_pkl": str(source_pkl),
    },
  }
  if include_raw_output:
    detail["output"] = _to_plain(output_root)
  if include_raw_meta:
    detail["meta"]["raw_meta"] = _to_plain(meta)
  return detail


def _score_value(row: Dict[str, Any]) -> float:
  value = _as_float(row.get("prediction_score_for_sampling", None))
  if value is None:
    return float("-inf")
  return float(value)


def _stable_seed(base_seed: int, cell_key: str) -> int:
  digest = hashlib.sha1(f"{base_seed}|{cell_key}".encode("utf-8")).hexdigest()[:8]
  return base_seed + int(digest, 16)


def _stable_row_key(split: str, row: Dict[str, Any], row_index: int) -> str:
  indice = row.get("indice", None)
  h5_path = row.get("h5_path", "")
  gt_label = row.get("gt_label", None)
  pred_label = row.get("pred_label", None)
  if isinstance(indice, int):
    return f"{split}|{gt_label}|{pred_label}|{indice}|{h5_path}"
  return f"{split}|{gt_label}|{pred_label}|{h5_path}|{row_index}"


def _collect_sampling_indices(
  rows: List[Dict[str, Any]],
  *,
  split: str,
  top_n: int,
  bottom_n: int,
  random_n: int,
  sample_random_seed: int,
) -> Tuple[Dict[int, Set[str]], int]:
  by_cell: Dict[Tuple[Optional[int], Optional[int]], List[int]] = defaultdict(list)
  for idx, row in enumerate(rows):
    by_cell[(row.get("gt_label", None), row.get("pred_label", None))].append(idx)

  selected_reasons: Dict[int, Set[str]] = defaultdict(set)
  seen_row_keys: Set[str] = set()
  for (gt_label, pred_label), indices in by_cell.items():
    sorted_indices = sorted(indices, key=lambda i: (-_score_value(rows[i]), i))
    top_indices = sorted_indices[: max(0, int(top_n))]
    bottom_indices = sorted_indices[-max(0, int(bottom_n)) :] if bottom_n > 0 else []

    cell_key = f"{split}|{gt_label}|{pred_label}"
    rng = random.Random(_stable_seed(sample_random_seed, cell_key))
    random_indices = list(indices)
    rng.shuffle(random_indices)
    random_indices = random_indices[: max(0, int(random_n))]

    for idx in top_indices:
      key = _stable_row_key(split, rows[idx], idx)
      if key not in seen_row_keys:
        seen_row_keys.add(key)
      selected_reasons[idx].add("top_score")

    for idx in bottom_indices:
      key = _stable_row_key(split, rows[idx], idx)
      if key not in seen_row_keys:
        seen_row_keys.add(key)
      selected_reasons[idx].add("bottom_score")

    for idx in random_indices:
      key = _stable_row_key(split, rows[idx], idx)
      if key not in seen_row_keys:
        seen_row_keys.add(key)
      selected_reasons[idx].add("random")

  return selected_reasons, len(by_cell)


def write_sample_analysis_bundle(
  *,
  work_dir: Path,
  analysis_result_files: Dict[str, Path],
  external_result_files: Optional[Dict[str, Path]] = None,
  class_names: Optional[List[str]] = None,
  secondary_class_names: Optional[List[str]] = None,
  dataset_info: Optional[Dict[str, Any]] = None,
  external_dataset_info: Optional[Dict[str, Any]] = None,
  out_name: str = "sample_analysis.xlsx",
  detail_dir_name: str = "sample_analysis_details",
  manifest_name: str = "sample_analysis_manifest.json",
  top_n: int = 4,
  bottom_n: int = 4,
  random_n: int = 8,
  json_write_workers: int = 8,
  sample_random_seed: int = 10482,
  clean_detail_dir: bool = False,
  show_progress: bool = False,
  include_raw_output: bool = False,
  include_raw_meta: bool = False,
  progress_name: str = "sample_analysis_export_progress.json",
  merge_existing_outputs: bool = True,
  export_detail_for: str = "sampled",
) -> Dict[str, Any]:
  export_detail_for = _normalize_export_detail_for(export_detail_for)
  start_all = time.perf_counter()
  work_dir = Path(work_dir).resolve()
  detail_dir = (work_dir / detail_dir_name).resolve()
  detail_dir.mkdir(parents=True, exist_ok=True)
  xlsx_path = (work_dir / out_name).resolve()

  if clean_detail_dir and detail_dir.exists():
    removed = 0
    for json_file in detail_dir.glob("*.json"):
      try:
        json_file.unlink()
        removed += 1
      except Exception:
        pass
    _log(f"[sample-analysis] cleaned detail dir: {detail_dir} removed_json={removed}")

  class_names = list(class_names or [])
  secondary_class_names = list(secondary_class_names or [])
  external_result_files = dict(external_result_files or {})
  dataset_info = dict(dataset_info or {})
  external_dataset_info = dict(external_dataset_info or {})

  def resolve_dataset_binding(split_name: str):
    if str(split_name).startswith("external_"):
      external_name = str(split_name)[len("external_"):]
      info = external_dataset_info.get(external_name, {})
      if not isinstance(info, dict):
        info = {}
      split_file = str(info.get("split_file", "") or "")
      return dict(
        dataset_split=split_name,
        dataset_source=split_file,
      )
    info = dataset_info.get(split_name, {})
    if not isinstance(info, dict):
      info = {}
    split_file = str(info.get("split_file", "") or "")
    return dict(
      dataset_split=split_name,
      dataset_source=split_file,
    )

  collect_sources_start = time.perf_counter()
  split_sources: List[Tuple[str, Path]] = []
  for split_name in ["train", "valid", "test"]:
    result_file = analysis_result_files.get(split_name, None)
    if result_file is None:
      continue
    result_path = Path(result_file)
    if result_path.exists():
      split_sources.append((split_name, result_path))
  for external_name, external_file in external_result_files.items():
    result_path = Path(external_file)
    if result_path.exists():
      split_sources.append((f"external_{external_name}", result_path))
  collect_sources_sec = time.perf_counter() - collect_sources_start

  rows: List[Dict[str, Any]] = []
  split_stats: List[Dict[str, Any]] = []
  total_json_write_fail = 0
  progress_path = (work_dir / progress_name).resolve()
  progress_state: Dict[str, Any] = {
    "schema_version": "sample_analysis_export_progress_v1",
    "run_dir": str(work_dir),
    "updated_at_unix": int(time.time()),
    "splits": {},
  }
  if merge_existing_outputs and progress_path.exists():
    try:
      existing_progress = json.loads(progress_path.read_text(encoding="utf-8"))
      if isinstance(existing_progress, dict):
        existing_splits = existing_progress.get("splits", {})
        if isinstance(existing_splits, dict):
          progress_state["splits"] = existing_splits
    except Exception:
      pass

  for split_name, result_path in split_sources:
    split_start = time.perf_counter()
    load_start = time.perf_counter()
    items = load_batch_items(result_path)
    load_sec = time.perf_counter() - load_start

    scan_start = time.perf_counter()
    split_rows: List[Dict[str, Any]] = []
    split_items: List[Dict[str, Any]] = []
    scan_iter = _progress(
      enumerate(items),
      total=len(items),
      desc=f"Scan {split_name}",
      enabled=show_progress,
      unit="sample",
    )
    for _, item in scan_iter:
      dataset_binding = resolve_dataset_binding(split_name)
      row = _build_row_summary(
        item,
        split=split_name,
        class_names=class_names,
        secondary_class_names=secondary_class_names,
        source_pkl=result_path,
        dataset_split=dataset_binding["dataset_split"],
        dataset_source=dataset_binding["dataset_source"],
        analysis_result_file=str(result_path),
      )
      split_rows.append(row)
      split_items.append(item)
    scan_sec = time.perf_counter() - scan_start

    selected_reasons, confusion_cell_count = _collect_sampling_indices(
      split_rows,
      split=split_name,
      top_n=top_n,
      bottom_n=bottom_n,
      random_n=random_n,
      sample_random_seed=sample_random_seed,
    )
    detail_indices = _detail_indices_for_export(
      export_detail_for=export_detail_for,
      num_rows=len(split_rows),
      selected_reasons=selected_reasons,
    )
    reason_counter = Counter()
    for reasons in selected_reasons.values():
      for reason in reasons:
        reason_counter[reason] += 1

    _log(
      "[sample-analysis] save JSON begin split={} mode={} dir={} detail_rows={} manifest_rows={} workers={} reasons={}".format(
        split_name,
        export_detail_for,
        detail_dir,
        len(detail_indices),
        len(split_rows),
        max(1, int(json_write_workers)),
        dict(reason_counter),
      )
    )

    json_start = time.perf_counter()
    json_written = 0
    json_failed = 0
    json_skipped = 0

    workers = max(1, int(json_write_workers))
    pbar = None
    if show_progress:
      try:
        from tqdm import tqdm

        pbar = tqdm(total=len(detail_indices), desc=f"Write JSON {split_name}", unit="file", leave=False)
      except Exception:
        pbar = None

    if workers <= 1:
      json_iter = _progress(
        detail_indices,
        total=len(detail_indices),
        desc=f"Write JSON {split_name}",
        enabled=False,
        unit="file",
      )
      for idx in json_iter:
        row = split_rows[idx]
        reasons = _reasons_for_detail_index(
          export_detail_for=export_detail_for,
          idx=idx,
          selected_reasons=selected_reasons,
        )
        file_name = _detail_filename(split_name, row)
        detail_path = detail_dir / file_name
        detail_rel = str(detail_path.relative_to(work_dir))
        existing = _find_existing_detail_json(detail_dir, work_dir, split_name, row)
        if existing is not None:
          _path, rel = existing
          row["detail_json"] = rel
          row["detail_json_generated"] = 1
          row["detail_json_sample_reason"] = ",".join(reasons)
          json_skipped += 1
          if pbar is not None:
            pbar.update(1)
          continue

        detail = _build_detail_record(
          split_items[idx],
          row=row,
          split=split_name,
          class_names=class_names,
          secondary_class_names=secondary_class_names,
          source_pkl=result_path,
          reasons=reasons,
          sample_random_seed=sample_random_seed,
          include_raw_output=include_raw_output,
          include_raw_meta=include_raw_meta,
        )
        try:
          _write_json_obj(detail_path, detail)
          row["detail_json"] = detail_rel
          row["detail_json_generated"] = 1
          row["detail_json_sample_reason"] = ",".join(reasons)
          json_written += 1
        except Exception as exc:
          row["detail_json"] = ""
          row["detail_json_generated"] = 0
          row["detail_json_sample_reason"] = ""
          json_failed += 1
          _log(
            "[sample-analysis] failed to save JSON run={} split={} path={} error={}".format(
              work_dir,
              split_name,
              detail_path,
              exc,
            )
          )
        if pbar is not None:
          pbar.update(1)
    else:
      with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {}

        def flush_futures(*, drain_all: bool = False):
          nonlocal json_written, json_failed
          while len(pending) > 0:
            if drain_all:
              done_set, _ = wait(set(pending.keys()))
            else:
              done_set, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
            for future in done_set:
              idx_done, detail_path_done, reasons_done, detail_rel_done = pending.pop(future)
              row_done = split_rows[idx_done]
              try:
                future.result()
                row_done["detail_json"] = detail_rel_done
                row_done["detail_json_generated"] = 1
                row_done["detail_json_sample_reason"] = ",".join(reasons_done)
                json_written += 1
              except Exception as exc:
                row_done["detail_json"] = ""
                row_done["detail_json_generated"] = 0
                row_done["detail_json_sample_reason"] = ""
                json_failed += 1
                _log(
                  "[sample-analysis] failed to save JSON run={} split={} path={} error={}".format(
                    work_dir,
                    split_name,
                    detail_path_done,
                    exc,
                  )
                )
              if pbar is not None:
                pbar.update(1)
            if not drain_all:
              return
          return

        for idx in detail_indices:
          row = split_rows[idx]
          reasons = _reasons_for_detail_index(
            export_detail_for=export_detail_for,
            idx=idx,
            selected_reasons=selected_reasons,
          )
          file_name = _detail_filename(split_name, row)
          detail_path = detail_dir / file_name
          detail_rel = str(detail_path.relative_to(work_dir))
          existing = _find_existing_detail_json(detail_dir, work_dir, split_name, row)
          if existing is not None:
            _path, rel = existing
            row["detail_json"] = rel
            row["detail_json_generated"] = 1
            row["detail_json_sample_reason"] = ",".join(reasons)
            json_skipped += 1
            if pbar is not None:
              pbar.update(1)
            continue

          detail = _build_detail_record(
            split_items[idx],
            row=row,
            split=split_name,
            class_names=class_names,
            secondary_class_names=secondary_class_names,
            source_pkl=result_path,
            reasons=reasons,
            sample_random_seed=sample_random_seed,
            include_raw_output=include_raw_output,
            include_raw_meta=include_raw_meta,
          )
          future = executor.submit(_write_json_obj, detail_path, detail)
          pending[future] = (idx, detail_path, reasons, detail_rel)
          if len(pending) >= workers * 2:
            flush_futures(drain_all=False)
        flush_futures(drain_all=True)
      if pbar is not None:
        pbar.close()

    json_write_sec = time.perf_counter() - json_start
    total_json_write_fail += json_failed

    for row in split_rows:
      row["h5_path"] = _as_rel_if_possible(str(row.get("h5_path", "")), base_dir=work_dir)
      row["source_pkl"] = _as_rel_if_possible(str(row.get("source_pkl", "")), base_dir=work_dir)
      rows.append(row)

    split_elapsed = time.perf_counter() - split_start
    row_count = len(split_rows)
    json_row_count = json_written
    rows_per_sec = 0.0 if split_elapsed <= 0 else float(row_count) / float(split_elapsed)
    detail_linked = int(
      sum(1 for row in split_rows if str(row.get("detail_json", "")).strip() != "")
    )
    split_stat = {
      "split": split_name,
      "source_pkl": str(result_path),
      "row_count": int(row_count),
      "json_row_count": int(json_row_count),
      "json_written_count": int(json_written),
      "json_failed_count": int(json_failed),
      "json_skipped_count": int(json_skipped),
      "json_completed_count": int(json_written + json_skipped),
      "json_detail_linked_count": int(detail_linked),
      "json_write_workers": int(workers),
      "confusion_cell_count": int(confusion_cell_count),
      "load_sec": float(load_sec),
      "scan_sec": float(scan_sec),
      "json_write_sec": float(json_write_sec),
      "elapsed_sec": float(split_elapsed),
      "rows_per_sec": float(rows_per_sec),
    }
    split_stats.append(split_stat)
    progress_state["splits"][split_name] = {
      "source_pkl": str(result_path),
      "expected_count": int(len(detail_indices)),
      "manifest_row_count": int(len(split_rows)),
      "export_detail_for": export_detail_for,
      "written_count": int(json_written),
      "skipped_count": int(json_skipped),
      "completed_count": int(json_written + json_skipped),
      "failed_count": int(json_failed),
      "json_detail_linked_count": int(detail_linked),
      "updated_at_unix": int(time.time()),
    }
    progress_state["updated_at_unix"] = int(time.time())
    _write_json_text(progress_path, json.dumps(progress_state, ensure_ascii=False, indent=2))
    _log(
      "[sample-analysis] save JSON end split={} wrote={} skipped={} failed={} elapsed={:.3f}s dir={}".format(
        split_name,
        json_written,
        json_skipped,
        json_failed,
        json_write_sec,
        detail_dir,
      )
    )

  dataframe_start = time.perf_counter()
  preferred_columns = [
    "h5_path",
    "dataset_split",
    "dataset_source",
    "analysis_result_file",
    "source_pkl",
    "detail_json_generated",
    "detail_json_sample_reason",
    "detail_json",
    "split",
    "indice",
    "confusion_gt_label",
    "confusion_pred_label",
    "confusion_cell_key",
    "prediction_score_for_sampling",
    "gt_label",
    "gt_label_name",
    "pred_label",
    "pred_label_name",
    "score",
    "gt_label_secondary",
    "gt_label_secondary_name",
    "pred_label_secondary",
    "pred_label_secondary_name",
    "score_secondary",
    "pred_label_secondary_conditional",
    "pred_label_secondary_conditional_name",
    "is_mismatch",
  ]
  if len(rows) == 0:
    current_df = pd.DataFrame(columns=preferred_columns)
  else:
    current_df = pd.DataFrame(rows)
    sorted_columns = [col for col in preferred_columns if col in current_df.columns]
    other_columns = [col for col in current_df.columns if col not in sorted_columns]
    current_df = current_df[sorted_columns + other_columns]
  df = current_df
  if merge_existing_outputs and xlsx_path.exists():
    try:
      existing_df = pd.read_excel(xlsx_path, sheet_name="manifest")
      split_names = [name for name, _ in split_sources]
      if isinstance(existing_df, pd.DataFrame) and len(existing_df) > 0:
        if "split" in existing_df.columns and len(split_names) > 0:
          existing_df = existing_df[~existing_df["split"].isin(split_names)]
        if len(df) == 0:
          df = existing_df
        else:
          df = pd.concat([existing_df, df], ignore_index=True)
    except Exception:
      pass
  dataframe_sec = time.perf_counter() - dataframe_start

  _log(
    "[sample-analysis] save Excel begin path={} rows={} cols={} sheet=manifest".format(
      xlsx_path,
      len(df),
      len(df.columns),
    )
  )
  excel_write_start = time.perf_counter()
  excel_progress = _progress(
    [0],
    total=1,
    desc="Save Excel",
    enabled=show_progress,
    unit="file",
  )
  for _ in excel_progress:
    df.to_excel(xlsx_path, index=False, sheet_name="manifest")
  excel_write_sec = time.perf_counter() - excel_write_start
  excel_size = xlsx_path.stat().st_size if xlsx_path.exists() else 0
  _log(
    "[sample-analysis] save Excel end path={} size={} elapsed={:.3f}s".format(
      xlsx_path,
      excel_size,
      excel_write_sec,
    )
  )

  total_sec = time.perf_counter() - start_all
  timings = {
    "collect_sources_sec": float(collect_sources_sec),
    "dataframe_sec": float(dataframe_sec),
    "excel_write_sec": float(excel_write_sec),
    "total_sec": float(total_sec),
  }
  slowest_stage = max(timings.items(), key=lambda kv: kv[1])[0] if len(timings) > 0 else ""
  _log(
    "[sample-analysis] run={} rows={} json_rows={} confusion_cells={} total={:.3f}s".format(
      work_dir,
      len(rows),
      sum(s.get("json_row_count", 0) for s in split_stats),
      sum(s.get("confusion_cell_count", 0) for s in split_stats),
      total_sec,
    )
  )
  _log("[sample-analysis] slowest_stage={} elapsed={:.3f}s".format(slowest_stage, timings.get(slowest_stage, 0.0)))

  manifest_path = (work_dir / manifest_name).resolve()
  merged_split_stats_map: Dict[str, Dict[str, Any]] = {}
  merged_analysis_result_files: Dict[str, str] = {}
  merged_external_result_files: Dict[str, str] = {}
  if merge_existing_outputs and manifest_path.exists():
    try:
      existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
      if isinstance(existing_manifest, dict):
        for old_item in existing_manifest.get("split_stats", []):
          if isinstance(old_item, dict):
            split_key = str(old_item.get("split", "")).strip()
            if split_key != "":
              merged_split_stats_map[split_key] = old_item
        old_analysis = existing_manifest.get("analysis_result_files", {})
        if isinstance(old_analysis, dict):
          merged_analysis_result_files.update({str(k): str(v) for k, v in old_analysis.items()})
        old_external = existing_manifest.get("external_result_files", {})
        if isinstance(old_external, dict):
          merged_external_result_files.update({str(k): str(v) for k, v in old_external.items()})
    except Exception:
      pass
  for new_item in split_stats:
    split_key = str(new_item.get("split", "")).strip()
    if split_key == "":
      continue
    old_item = merged_split_stats_map.get(split_key)
    if isinstance(old_item, dict):
      add_w = int(new_item.get("json_written_count", new_item.get("json_row_count", 0)) or 0)
      old_w = int(old_item.get("json_written_count", old_item.get("json_row_count", 0)) or 0)
      merged_item = dict(new_item)
      if add_w == 0:
        merged_item["json_written_count"] = old_w
        merged_item["json_row_count"] = old_w
      else:
        merged_item["json_written_count"] = old_w + add_w
        merged_item["json_row_count"] = old_w + add_w
      merged_split_stats_map[split_key] = merged_item
    else:
      merged_split_stats_map[split_key] = new_item
  merged_split_stats = sorted(list(merged_split_stats_map.values()), key=lambda x: str(x.get("split", "")))
  linked_by_split = _json_detail_linked_counts_from_manifest_df(df)
  merged_split_stats = [
    {
      **dict(item),
      "json_detail_linked_count": int(
        linked_by_split.get(str(item.get("split", "")).strip(), int(item.get("json_detail_linked_count", 0) or 0))
      ),
    }
    for item in merged_split_stats
  ]
  merged_analysis_result_files.update({
    str(k): str(v)
    for k, v in dict(analysis_result_files or {}).items()
  })
  merged_external_result_files.update({
    str(k): str(v)
    for k, v in external_result_files.items()
  })
  merged_json_failed_count = int(sum(s.get("json_failed_count", 0) for s in merged_split_stats))
  total_detail_linked = int(sum(linked_by_split.values()))
  manifest_payload = {
    "schema_version": "sample_analysis_manifest_v1",
    "run_dir": str(work_dir),
    "xlsx_file": str(xlsx_path),
    "detail_dir": str(detail_dir),
    "run_signature": str(work_dir.name),
    "class_names": class_names,
    "secondary_class_names": secondary_class_names,
    "analysis_result_files": merged_analysis_result_files,
    "external_result_files": merged_external_result_files,
    "dataset_info": _to_plain(dataset_info),
    "external_dataset_info": _to_plain(external_dataset_info),
    "split_stats": merged_split_stats,
    "progress_file": str(progress_path),
    "timings": timings,
    "row_count": int(len(df)),
    "json_row_count": int(sum(s.get("json_row_count", 0) for s in merged_split_stats)),
    "json_written_count": int(sum(s.get("json_written_count", 0) for s in merged_split_stats)),
    "json_skipped_count": int(sum(s.get("json_skipped_count", 0) for s in merged_split_stats)),
    "json_completed_count": int(sum(s.get("json_completed_count", 0) for s in merged_split_stats)),
    "json_failed_count": merged_json_failed_count,
    "json_detail_linked_count": total_detail_linked,
    "json_detail_linked_by_split": dict(sorted(linked_by_split.items(), key=lambda kv: kv[0])),
    "include_raw_output": bool(include_raw_output),
    "include_raw_meta": bool(include_raw_meta),
    "export_detail_for": export_detail_for,
  }
  _write_json_text(manifest_path, json.dumps(manifest_payload, ensure_ascii=False, indent=2))

  return {
    "xlsx_file": str(xlsx_path),
    "detail_dir": str(detail_dir),
    "manifest_file": str(manifest_path),
    "row_count": int(len(df)),
    "json_row_count": int(sum(s.get("json_row_count", 0) for s in merged_split_stats)),
    "json_written_count": int(sum(s.get("json_written_count", 0) for s in merged_split_stats)),
    "json_skipped_count": int(sum(s.get("json_skipped_count", 0) for s in merged_split_stats)),
    "json_completed_count": int(sum(s.get("json_completed_count", 0) for s in merged_split_stats)),
    "json_failed_count": merged_json_failed_count,
    "json_detail_linked_count": total_detail_linked,
    "json_detail_linked_by_split": dict(sorted(linked_by_split.items(), key=lambda kv: kv[0])),
    "split_stats": merged_split_stats,
    "timings": timings,
    "progress_file": str(progress_path),
  }
