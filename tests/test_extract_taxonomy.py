from mmengine.config import Config

from slimai.export.bundle import build_secondary_head_keys, extract_taxonomy


def test_build_secondary_head_keys_from_canonical_mapping():
  cfg = Config(
    dict(
      PRIMARY_HEAD_KEYS=["p0", "p1"],
      SECONDARY_CANONICAL_LOCAL_MAPPING=dict(
        p0=dict(b=1, a=0),
        p1=dict(c=0),
      ),
    )
  )
  assert build_secondary_head_keys(cfg) == ["a", "b", "c"]


def test_extract_taxonomy_includes_secondary_names():
  cfg = Config(
    dict(
      PRIMARY_HEAD_KEYS=["良性病变", "良性肿瘤"],
      SECONDARY_CANONICAL_LOCAL_MAPPING=dict(
        良性病变=dict(普通导管增生=0, 柱状细胞病变=1),
        良性肿瘤=dict(纤维腺瘤=0),
      ),
      SECONDARY_GLOBAL_PARENT_IDX=[0, 0, 1],
      SECONDARY_GLOBAL_LOCAL_IDX=[0, 1, 0],
      NUM_CLASSES=2,
      EMBEDDING_DIM=1024,
      PATCH_ENCODER_NAME="UNI",
      TARGET_NAME="BREXI",
    )
  )
  taxonomy = extract_taxonomy(cfg)
  assert taxonomy["SECONDARY_HEAD_KEYS"] == ["普通导管增生", "柱状细胞病变", "纤维腺瘤"]
  assert taxonomy["NUM_SECONDARY_CLASSES"] == 3
  assert taxonomy["SECONDARY_CANONICAL_LOCAL_MAPPING"]["良性病变"]["普通导管增生"] == 0
