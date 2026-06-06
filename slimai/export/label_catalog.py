from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml
from mmengine.config import Config

from slimai.export.manifest import file_md5

DEFAULT_LABEL_CATALOG_FILENAME = "label_catalog.yaml"


def load_label_catalog(path: Path | str) -> Dict[str, Any]:
  catalog_path = Path(path)
  if not catalog_path.is_file():
    raise FileNotFoundError(f"Label catalog not found: {catalog_path}")
  payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise ValueError(f"Label catalog must be a mapping, got {type(payload).__name__}")
  return payload


def resolve_label_catalog_path(cfg: Config) -> Optional[Path]:
  catalog_file = getattr(cfg, "LABEL_CATALOG_FILE", None)
  if not catalog_file:
    return None
  catalog_path = Path(str(catalog_file))
  if not catalog_path.is_file():
    raise FileNotFoundError(f"LABEL_CATALOG_FILE not found: {catalog_path}")
  return catalog_path


def _ordered_secondary_names(
  primary_head_keys: Sequence[str],
  secondary_canonical_local_mapping: Mapping[str, Mapping[str, int]],
) -> List[str]:
  names: List[str] = []
  for primary_name in primary_head_keys:
    local_mapping = secondary_canonical_local_mapping[primary_name]
    ordered = sorted(local_mapping.items(), key=lambda item: int(item[1]))
    names.extend(str(name) for name, _ in ordered)
  return names


def validate_label_catalog(
  catalog: Mapping[str, Any],
  *,
  primary_head_keys: Sequence[str],
  secondary_canonical_local_mapping: Mapping[str, Mapping[str, int]],
) -> None:
  classes = catalog.get("classes")
  if not isinstance(classes, dict) or len(classes) == 0:
    raise ValueError("Label catalog must contain non-empty 'classes'")

  missing_primary = [name for name in primary_head_keys if name not in classes]
  if missing_primary:
    raise ValueError(f"Label catalog missing primary classes: {missing_primary}")

  extra_primary = [name for name in classes.keys() if name not in primary_head_keys]
  if extra_primary:
    raise ValueError(f"Label catalog has unexpected primary classes: {extra_primary}")

  for primary_name in primary_head_keys:
    primary_entry = classes[primary_name]
    if not isinstance(primary_entry, dict):
      raise ValueError(f"Primary class entry must be a mapping: {primary_name}")
    for field in ("en", "abbrev"):
      if field not in primary_entry:
        raise ValueError(f"Primary class missing '{field}': {primary_name}")

    secondary_block = primary_entry.get("secondary", {})
    if not isinstance(secondary_block, dict):
      raise ValueError(f"Primary class secondary must be a mapping: {primary_name}")

    expected_secondary = _ordered_secondary_names([primary_name], secondary_canonical_local_mapping)
    missing_secondary = [name for name in expected_secondary if name not in secondary_block]
    if missing_secondary:
      raise ValueError(
        f"Label catalog missing secondary classes under {primary_name}: {missing_secondary}"
      )

    extra_secondary = [name for name in secondary_block.keys() if name not in expected_secondary]
    if extra_secondary:
      raise ValueError(
        f"Label catalog has unexpected secondary classes under {primary_name}: {extra_secondary}"
      )

    for secondary_name in expected_secondary:
      secondary_entry = secondary_block[secondary_name]
      if not isinstance(secondary_entry, dict):
        raise ValueError(f"Secondary class entry must be a mapping: {primary_name}/{secondary_name}")
      for field in ("en", "abbrev"):
        if field not in secondary_entry:
          raise ValueError(
            f"Secondary class missing '{field}': {primary_name}/{secondary_name}"
          )

      tertiary_block = secondary_entry.get("tertiary")
      if tertiary_block is None:
        continue
      if not isinstance(tertiary_block, dict):
        raise ValueError(f"Tertiary block must be a mapping: {primary_name}/{secondary_name}")
      for tertiary_name, tertiary_entry in tertiary_block.items():
        if not isinstance(tertiary_entry, dict):
          raise ValueError(
            f"Tertiary class entry must be a mapping: {primary_name}/{secondary_name}/{tertiary_name}"
          )
        for field in ("en", "abbrev"):
          if field not in tertiary_entry:
            raise ValueError(
              f"Tertiary class missing '{field}': {primary_name}/{secondary_name}/{tertiary_name}"
            )
  return


def flatten_primary_labels(
  catalog: Mapping[str, Any],
  primary_head_keys: Sequence[str],
) -> Dict[str, List[str]]:
  classes = catalog["classes"]
  return dict(
    PRIMARY_EN=[classes[name]["en"] for name in primary_head_keys],
    PRIMARY_ABBREV=[classes[name]["abbrev"] for name in primary_head_keys],
  )


def flatten_secondary_labels(
  catalog: Mapping[str, Any],
  *,
  primary_head_keys: Sequence[str],
  secondary_canonical_local_mapping: Mapping[str, Mapping[str, int]],
) -> Dict[str, List[str]]:
  classes = catalog["classes"]
  secondary_en: List[str] = []
  secondary_abbrev: List[str] = []
  for primary_name in primary_head_keys:
    secondary_block = classes[primary_name]["secondary"]
    for secondary_name in _ordered_secondary_names([primary_name], secondary_canonical_local_mapping):
      entry = secondary_block[secondary_name]
      secondary_en.append(entry["en"])
      secondary_abbrev.append(entry["abbrev"])
  return dict(SECONDARY_EN=secondary_en, SECONDARY_ABBREV=secondary_abbrev)


def flatten_platform_fields(catalog: Mapping[str, Any]) -> Dict[str, Any]:
  platform = catalog.get("platform", {})
  if not isinstance(platform, dict):
    return {}
  fields: Dict[str, Any] = {}
  default_negative_en = platform.get("default_negative_en")
  if default_negative_en is not None:
    fields["PLATFORM_DEFAULT_NEGATIVE"] = str(default_negative_en)
    fields["PLATFORM_DEFAULT_NEGATIVE_EN"] = str(default_negative_en)
  binary_positive_indices = platform.get("binary_positive_indices")
  if binary_positive_indices is not None:
    fields["BINARY_POSITIVE_INDICES"] = list(binary_positive_indices)
  return fields


def attach_label_catalog_to_taxonomy(
  taxonomy: Dict[str, Any],
  cfg: Config,
  *,
  output_dir: Path,
  catalog_filename: str = DEFAULT_LABEL_CATALOG_FILENAME,
) -> Dict[str, Any]:
  catalog_path = resolve_label_catalog_path(cfg)
  if catalog_path is None:
    return taxonomy

  catalog = load_label_catalog(catalog_path)
  primary_head_keys = taxonomy.get("PRIMARY_HEAD_KEYS") or list(getattr(cfg, "PRIMARY_HEAD_KEYS", []))
  secondary_canonical_local_mapping = taxonomy.get("SECONDARY_CANONICAL_LOCAL_MAPPING")
  if secondary_canonical_local_mapping is None:
    secondary_canonical_local_mapping = getattr(cfg, "SECONDARY_CANONICAL_LOCAL_MAPPING", {})

  validate_label_catalog(
    catalog,
    primary_head_keys=primary_head_keys,
    secondary_canonical_local_mapping=secondary_canonical_local_mapping,
  )

  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  dst_path = output_dir / catalog_filename
  shutil.copy2(catalog_path, dst_path)

  enriched = dict(taxonomy)
  enriched["label_catalog_file"] = catalog_filename
  enriched["label_catalog_md5"] = file_md5(dst_path)
  version = catalog.get("version")
  if version is not None:
    enriched["label_catalog_version"] = str(version)
  enriched.update(flatten_platform_fields(catalog))
  return enriched
