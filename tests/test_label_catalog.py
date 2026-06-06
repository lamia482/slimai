from pathlib import Path

import pytest

from slimai.export.label_catalog import (
  attach_label_catalog_to_taxonomy,
  flatten_secondary_labels,
  load_label_catalog,
  validate_label_catalog,
)


CATALOG_PATH = Path("/hzztai/wangqiang/workspace/projects/brexi/taxonomy/label_catalog.yaml")

PRIMARY_HEAD_KEYS = ["良性病变", "良性肿瘤", "癌前病变", "恶性肿瘤（癌）"]
SECONDARY_CANONICAL_LOCAL_MAPPING = {
  "良性病变": {
    "普通导管增生": 0,
    "柱状细胞病变": 1,
    "硬化性腺病": 2,
    "大汗腺腺病和腺瘤": 3,
    "其他良性病变": 4,
  },
  "良性肿瘤": {
    "管状腺瘤": 0,
    "导管腺瘤": 1,
    "纤维腺瘤": 2,
    "乳头状瘤（导管内乳头状瘤）": 3,
    "其他良性肿瘤": 4,
  },
  "癌前病变": {
    "异型导管上皮增生": 0,
    "非典型小叶增生": 1,
    "其他癌前病变": 2,
  },
  "恶性肿瘤（癌）": {
    "原位癌": 0,
    "乳头状癌": 1,
    "浸润性癌": 2,
    "其他恶性肿瘤（癌）": 3,
  },
}


def test_load_and_validate_label_catalog():
  catalog = load_label_catalog(CATALOG_PATH)
  validate_label_catalog(
    catalog,
    primary_head_keys=PRIMARY_HEAD_KEYS,
    secondary_canonical_local_mapping=SECONDARY_CANONICAL_LOCAL_MAPPING,
  )


def test_flatten_secondary_labels_matches_head_order():
  catalog = load_label_catalog(CATALOG_PATH)
  flattened = flatten_secondary_labels(
    catalog,
    primary_head_keys=PRIMARY_HEAD_KEYS,
    secondary_canonical_local_mapping=SECONDARY_CANONICAL_LOCAL_MAPPING,
  )
  assert flattened["SECONDARY_ABBREV"][-1] == "OMT"
  assert flattened["SECONDARY_EN"][0] == "Usual Ductal Hyperplasia"


def test_attach_label_catalog_to_taxonomy(tmp_path):
  pytest.importorskip("mmengine")
  from mmengine.config import Config

  cfg = Config(
    dict(
      LABEL_CATALOG_FILE=str(CATALOG_PATH),
      PRIMARY_HEAD_KEYS=PRIMARY_HEAD_KEYS,
      SECONDARY_CANONICAL_LOCAL_MAPPING=SECONDARY_CANONICAL_LOCAL_MAPPING,
    )
  )
  taxonomy = dict(
    PRIMARY_HEAD_KEYS=PRIMARY_HEAD_KEYS,
    SECONDARY_HEAD_KEYS=[
      "普通导管增生", "柱状细胞病变", "硬化性腺病", "大汗腺腺病和腺瘤", "其他良性病变",
      "管状腺瘤", "导管腺瘤", "纤维腺瘤", "乳头状瘤（导管内乳头状瘤）", "其他良性肿瘤",
      "异型导管上皮增生", "非典型小叶增生", "其他癌前病变",
      "原位癌", "乳头状癌", "浸润性癌", "其他恶性肿瘤（癌）",
    ],
    SECONDARY_CANONICAL_LOCAL_MAPPING=SECONDARY_CANONICAL_LOCAL_MAPPING,
  )
  enriched = attach_label_catalog_to_taxonomy(taxonomy, cfg, output_dir=tmp_path)
  assert enriched["label_catalog_file"] == "label_catalog.yaml"
  assert (tmp_path / "label_catalog.yaml").is_file()
  assert enriched["PLATFORM_DEFAULT_NEGATIVE_EN"] == "Other"
  assert enriched["BINARY_POSITIVE_INDICES"] == [2, 3]
  assert "PRIMARY_EN" not in enriched
  assert "SECONDARY_EN" not in enriched
