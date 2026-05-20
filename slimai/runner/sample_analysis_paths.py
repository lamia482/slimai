"""Discover analysis PKL paths and class-name lists for sample_analysis export."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import mmengine


def mapping_to_class_names(mapping: Any) -> List[str]:
  if not isinstance(mapping, dict) or len(mapping) == 0:
    return []
  max_index = -1
  for _, index in mapping.items():
    if isinstance(index, int):
      max_index = max(max_index, int(index))
  if max_index < 0:
    return []
  names = [str(i) for i in range(max_index + 1)]
  for name, index in mapping.items():
    if isinstance(index, int) and 0 <= int(index) < len(names):
      names[int(index)] = str(name)
  return names


def _pick_latest_by_mtime(files: List[Path]) -> Optional[Path]:
  if len(files) == 0:
    return None
  return max(files, key=lambda p: p.stat().st_mtime).resolve()


def parse_split_filter(value: str) -> Set[str]:
  out: Set[str] = set()
  for token in str(value or "").split(","):
    text = token.strip()
    if text == "":
      continue
    out.add(text)
  return out


def load_class_names_from_taxonomy(run_dir: Path) -> Tuple[List[str], List[str]]:
  taxonomy_file = Path(run_dir) / "label_taxonomy.json"
  if not taxonomy_file.exists():
    return [], []
  try:
    payload = json.loads(taxonomy_file.read_text(encoding="utf-8"))
  except Exception:
    return [], []

  primary_mapping = payload.get("PRIMARY_LABEL_MAPPING")
  if not isinstance(primary_mapping, dict):
    primary_mapping = (payload.get("LABEL_TAXONOMY", {}) or {}).get("primary", {})
  secondary_mapping = payload.get("SECONDARY_LABEL_MAPPING")
  if not isinstance(secondary_mapping, dict):
    secondary_mapping = (payload.get("LABEL_TAXONOMY", {}) or {}).get("secondary_label_mapping", {})
  if not isinstance(primary_mapping, dict):
    primary_mapping = {}
  if not isinstance(secondary_mapping, dict):
    secondary_mapping = {}
  return mapping_to_class_names(primary_mapping), mapping_to_class_names(secondary_mapping)


def load_class_names_from_config(config_path: Path) -> Tuple[List[str], List[str]]:
  cfg = mmengine.Config.fromfile(str(config_path))
  return (
    mapping_to_class_names(cfg.get("LABEL_MAPPING", {})),
    mapping_to_class_names(cfg.get("SECONDARY_LABEL_MAPPING", {})),
  )


def load_class_names(run_dir: Path) -> Tuple[List[str], List[str]]:
  """Taxonomy first; fallback to run_dir/config.py when taxonomy is missing or empty."""
  run_dir = Path(run_dir).resolve()
  primary, secondary = load_class_names_from_taxonomy(run_dir)
  if len(primary) > 0 or len(secondary) > 0:
    return primary, secondary
  config_file = run_dir / "config.py"
  if config_file.exists():
    return load_class_names_from_config(config_file)
  return [], []


def discover_analysis_pkls(
  results_dir: Path,
  *,
  allowed_splits: Optional[Set[str]] = None,
) -> Tuple[Dict[str, Path], Dict[str, Path]]:
  """Pick latest analysis PKL per split by file mtime."""
  results_dir = Path(results_dir).resolve()
  allowed = allowed_splits if allowed_splits is not None else set()
  filter_splits = len(allowed) > 0

  analysis_result_files: Dict[str, Path] = {}
  external_result_files: Dict[str, Path] = {}

  split_prefixes = {
    "train": "analysis_train_best_epoch_",
    "valid": "analysis_valid_best_epoch_",
    "test": "analysis_test_best_epoch_",
  }
  for split_name, prefix in split_prefixes.items():
    if filter_splits and split_name not in allowed:
      continue
    candidates = list(results_dir.glob(f"{prefix}*.pkl"))
    selected = _pick_latest_by_mtime(candidates)
    if selected is not None:
      analysis_result_files[split_name] = selected

  for path in results_dir.glob("analysis_external_*_best_epoch_*.pkl"):
    stem = path.stem
    prefix = "analysis_external_"
    middle = "_best_epoch_"
    if not stem.startswith(prefix) or middle not in stem:
      continue
    external_name = stem[len(prefix):].split(middle)[0].strip()
    if external_name == "":
      continue
    full_split_name = f"external_{external_name}"
    if filter_splits and full_split_name not in allowed:
      continue
    current = external_result_files.get(external_name)
    if current is None or path.stat().st_mtime > current.stat().st_mtime:
      external_result_files[external_name] = path.resolve()

  return analysis_result_files, external_result_files
