from slimai.runner.report import ExperimentReporter


def _build_reporter():
  cfg = dict(
    PRIMARY_HEAD_KEYS=["良性病变", "良性肿瘤", "癌前病变", "恶性肿瘤（癌）"],
    SECONDARY_CANONICAL_LOCAL_MAPPING={
      "良性病变": {
        "普通导管增生": 0,
        "其他良性病变": 4,
      },
      "良性肿瘤": {
        "纤维腺瘤": 2,
        "其他良性肿瘤": 4,
      },
      "癌前病变": {
        "其他癌前病变": 2,
      },
      "恶性肿瘤（癌）": {
        "浸润性癌": 2,
        "其他恶性肿瘤（癌）": 3,
      },
    },
    SECONDARY_ALIAS_TO_CANONICAL={
      "良性病变": {"化生性病变": "其他良性病变"},
      "良性肿瘤": {"间叶源性肿瘤": "其他良性肿瘤"},
      "癌前病变": {},
      "恶性肿瘤（癌）": {"转移癌": "其他恶性肿瘤（癌）"},
    },
    VALID_LOADER=dict(dataset=dict()),
  )
  reporter = ExperimentReporter.__new__(ExperimentReporter)
  reporter.cfg = cfg
  reporter.secondary_class_names = reporter._resolve_secondary_canonical_class_names_from_cfg()
  reporter.display_secondary_class_names = reporter._resolve_display_class_names(reporter.secondary_class_names)
  reporter.secondary_alias_to_canonical = reporter._resolve_secondary_alias_to_canonical_flat()
  reporter.chart_counter = 0
  return reporter


def test_secondary_canonical_class_names_exclude_alias_overwrite():
  reporter = _build_reporter()
  assert reporter.secondary_class_names == [
    "普通导管增生",
    "其他良性病变",
    "纤维腺瘤",
    "其他良性肿瘤",
    "其他癌前病变",
    "浸润性癌",
    "其他恶性肿瘤（癌）",
  ]


def test_secondary_alias_confusion_matrix_rows_and_cols():
  reporter = _build_reporter()
  canonical = reporter.secondary_class_names
  other_benign_idx = canonical.index("其他良性病变")
  samples = [
    dict(label=other_benign_idx, pred=other_benign_idx, prob=[0.1] * len(canonical), label_secondary_name="化生性病变"),
    dict(label=other_benign_idx, pred=other_benign_idx, prob=[0.1] * len(canonical), label_secondary_name="其他良性病变"),
  ]
  html = reporter._chart_secondary_alias_confusion_matrix(samples, reporter.display_secondary_class_names)
  assert "其他良性病变-化生性病变" in html
  assert "其他良性病变" in html
  assert "Confusion Matrix (GT: alias前 / Pred: canonical)" in html
