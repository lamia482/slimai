import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from slimai.helper.help_build import SOURCES
from slimai.helper.help_utils import print_log


@SOURCES.register_module()
class SheetSource(object):
  def __init__(self, sheet_file, *, 
               col_mapping, sheet_name=None, 
               filter=None, apply=None):
    self.sheet_file = sheet_file
    self.col_mapping = col_mapping
    self.sheet_name = sheet_name
    assert (
      isinstance(filter, (tuple, list)) and 
      all(isinstance(item, (tuple, list)) for item in filter)
    ) or filter is None, (
      "filter must be a tuple or list of tuples or None, but got {filter}"
    )
    if filter is not None:
      self.filters = [
        (key, op, value) for key, op, value in filter
      ]
    else:
      self.filters = None
    assert (
      isinstance(apply, (tuple, list)) and 
      all(isinstance(item, (tuple, list)) for item in apply)
    ) or apply is None, (
      "apply must be a tuple or list of tuples or None, but got {apply}"
    )
    if apply is not None:
      self.applies = [
        (key, func) for key, func in apply
      ]
    else:
      self.applies = None
    return

  def __call__(self) -> Dict[str, List[str]]:
    if self.sheet_file.endswith(".xlsx"):
      read_fn = pd.read_excel
    elif self.sheet_file.endswith(".csv"):
      read_fn = pd.read_csv
    else:
      raise ValueError(f"Unsupported file extension: {self.sheet_file.split('.')[-1]}")

    if self.sheet_name is None:
      sheet_name = None
    elif isinstance(self.sheet_name, (tuple, list)):
      sheet_name = self.sheet_name
    elif isinstance(self.sheet_name, str):
      sheet_name = [self.sheet_name]
    
    df_dict = read_fn(self.sheet_file, sheet_name=sheet_name) # type: ignore
    df_list = []
    col_mapping = self.col_mapping.copy()
    if "_SHEET_NAME_" not in col_mapping:
      col_mapping["_SHEET_NAME_"] = "sheet_name"
    for sheet_name, df in df_dict.items():
      df.loc[:, ["_SHEET_NAME_"]] = sheet_name
      df = df[list(col_mapping.keys())]
      df.columns = list(col_mapping.values())
      df_list.append(df)

    df = pd.concat(df_list)
    if self.filters is not None:
      for key, op, value in self.filters:
        df = df[df[key].apply(lambda x: eval(f"'{x}' {op} '{value}'"))]
    if self.applies is not None:
      for key, func in self.applies:
        df[key] = df[key].apply(eval(func))
    return df.to_dict(orient="list") # type: ignore


@SOURCES.register_module()
class StratifiedSheetSource(object):
  def __init__(
    self,
    sheet_file,
    *,
    sheet_name=None,
    file_col="EMBEDDING",
    label_col="一级分类",
    secondary_label_col: Optional[str] = None,
    split_col="split",
    output_split_file: Optional[str] = None,
    random_seed: int = 10482,
    split_ratio: Optional[Dict[str, float]] = None,
    label_mapping: Optional[Dict[str, int]] = None,
    secondary_label_mapping: Optional[Dict[str, int]] = None,
    secondary_local_mapping: Optional[Dict[str, Dict[str, int]]] = None,
    secondary_label_required: bool = True,
    stratify_label_col: str = "primary",
    path_mapping: Optional[Sequence[Tuple[str, str]]] = None,
    mapped_file_col: str = "EMBEDDING_MAPPED",
    case_col: str = "案例号",
    center_col: str = "中心",
    diag_output_file: Optional[str] = None,
    min_class_samples: int = 1,
  ):
    self.sheet_file = sheet_file
    self.sheet_name = sheet_name
    self.file_col = file_col
    self.label_col = label_col
    self.secondary_label_col = secondary_label_col
    self.split_col = split_col
    self.output_split_file = output_split_file
    self.random_seed = random_seed
    self.split_ratio = split_ratio or dict(train=0.8, valid=0.1, test=0.1)
    self.label_mapping = label_mapping
    self.secondary_label_mapping = secondary_label_mapping
    self.secondary_local_mapping = secondary_local_mapping or {}
    self.secondary_label_required = bool(secondary_label_required)
    self.stratify_label_col = stratify_label_col
    self.path_mapping = list(path_mapping or [])
    self.mapped_file_col = mapped_file_col
    self.case_col = case_col
    self.center_col = center_col
    self.diag_output_file = diag_output_file
    self.min_class_samples = int(max(min_class_samples, 1))
    return

  def map_path(self, path):
    path = str(path)
    for src_prefix, dst_prefix in self.path_mapping:
      if path.startswith(src_prefix):
        return dst_prefix + path[len(src_prefix) :]
      path = path.replace(src_prefix, dst_prefix, 1)
    return path

  def _read_sheet_from(self, sheet_file, sheet_name=None):
    source = SheetSource(
      sheet_file=sheet_file,
      col_mapping={
        self.file_col: "raw_file",
        self.label_col: "label_name",
        **(
          {self.secondary_label_col: "label_secondary_name"}
          if isinstance(self.secondary_label_col, str)
          else {}
        ),
      },
      sheet_name=self.sheet_name if sheet_name is None else sheet_name,
      filter=None,
      apply=None,
    )
    data = source()
    df = pd.DataFrame(data)
    return df

  def _read_sheet(self):
    return self._read_sheet_from(self.sheet_file, self.sheet_name)

  def _prepare_dataframe(self, df: Optional[pd.DataFrame] = None, *, center_name: Optional[str] = None, center_col: str = "center"):
    df = (self._read_sheet() if df is None else df).copy()
    if center_name is not None:
      df[center_col] = center_name

    df["label_idx"] = (
      df["label_name"].map(self.label_mapping)
      if self.label_mapping is not None
      else df["label_name"]
    )
    has_secondary = (
      isinstance(self.secondary_label_col, str)
      and self.secondary_label_col != ""
      and "label_secondary_name" in df.columns
    )
    if has_secondary:
      if self.secondary_label_mapping is not None:
        df["label_secondary_idx"] = df["label_secondary_name"].map(self.secondary_label_mapping)
      else:
        df["label_secondary_idx"] = df["label_secondary_name"]
      df["label_secondary_local"] = df.apply(
        lambda row: self._map_secondary_local(row.get("label_name", None), row.get("label_secondary_name", None)),
        axis=1,
      )
    else:
      df["label_secondary_name"] = None
      df["label_secondary_idx"] = None
      df["label_secondary_local"] = None
    df[self.mapped_file_col] = df["raw_file"].apply(
      lambda x: self.map_path(x) if pd.notna(x) else x
    )
    df[self.split_col] = "ignored"
    df["ignore_reason"] = ""

    valid_mask = (
      df["raw_file"].notna()
      & df["label_idx"].notna()
      & df[self.mapped_file_col].notna()
    )
    if has_secondary and self.secondary_label_required:
      valid_mask = (
        valid_mask
        & df["label_secondary_idx"].notna()
        & df["label_secondary_local"].notna()
      )
    df.loc[df["raw_file"].isna(), "ignore_reason"] = "missing_embedding"
    df.loc[df["label_idx"].isna(), "ignore_reason"] = "unknown_label"
    df.loc[df[self.mapped_file_col].isna(), "ignore_reason"] = "invalid_mapped_path"
    if has_secondary and self.secondary_label_required:
      df.loc[df["label_secondary_idx"].isna(), "ignore_reason"] = "unknown_secondary_label"
      df.loc[df["label_secondary_local"].isna(), "ignore_reason"] = "invalid_secondary_local"

    usable_df = df[valid_mask].copy()
    return df, usable_df, valid_mask

  def _map_secondary_local(self, label_name, label_secondary_name):
    if label_name is None or label_secondary_name is None:
      return None
    parent = str(label_name)
    child = str(label_secondary_name)
    local_mapping = self.secondary_local_mapping.get(parent, None)
    if local_mapping is None:
      return None
    if child not in local_mapping:
      return None
    return int(local_mapping[child])

  def _validate_ratio(self):
    train_ratio = self.split_ratio.get("train", 0.0)
    valid_ratio = self.split_ratio.get("valid", 0.0)
    test_ratio = self.split_ratio.get("test", 0.0)
    ratio_sum = train_ratio + valid_ratio + test_ratio
    assert abs(ratio_sum - 1.0) < 1e-8, (
      f"split_ratio must sum to 1.0, but got {self.split_ratio}"
    )
    return train_ratio, valid_ratio, test_ratio

  def _dump_split_file(self, df: pd.DataFrame):
    if self.output_split_file is None:
      return None
    output_path = Path(self.output_split_file)
    os.makedirs(output_path.parent, exist_ok=True)
    output_sheet_name = (
      f"{self.sheet_name}_split"
      if isinstance(self.sheet_name, str)
      else "split"
    )
    df.to_excel(output_path, sheet_name=output_sheet_name, index=False)
    return output_path.as_posix()

  def _to_records(self, df: pd.DataFrame):
    has_secondary = (
      isinstance(self.secondary_label_col, str)
      and self.secondary_label_col != ""
      and "label_secondary_name" in df.columns
    )
    if not has_secondary:
      return list(zip(df[self.mapped_file_col].tolist(), df["label_name"].tolist()))

    records = []
    for _, row in df.iterrows():
      label_secondary_name = row.get("label_secondary_name", None)
      if pd.isna(label_secondary_name):
        label_secondary_name = None
      label_secondary_local = row.get("label_secondary_local", None)
      if label_secondary_local is None or pd.isna(label_secondary_local):
        label_secondary_local = -1
      records.append(
        (
          row[self.mapped_file_col],
          dict(
            label=row["label_name"],
            label_secondary=label_secondary_name,
            label_secondary_local=int(label_secondary_local),
          ),
        )
      )
    return records

  def _resolve_stratify_column(self, usable_df: pd.DataFrame) -> pd.Series:
    if self.stratify_label_col in ["primary", "label", "一级分类"]:
      return usable_df["label_name"]
    if self.stratify_label_col in ["secondary", "label_secondary", "二级分类"] and "label_secondary_name" in usable_df.columns:
      return usable_df["label_secondary_name"]
    if self.stratify_label_col in usable_df.columns:
      return usable_df[self.stratify_label_col]
    return usable_df["label_name"]

  def _compute_split_diag(
    self,
    df: pd.DataFrame,
    *,
    train_df: Optional[pd.DataFrame] = None,
    valid_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
  ) -> Dict[str, Any]:
    diag: Dict[str, Any] = {}
    split_frames = {
      "train": train_df if train_df is not None else df[df[self.split_col] == "train"],
      "valid": valid_df if valid_df is not None else df[df[self.split_col] == "valid"],
      "test": test_df if test_df is not None else df[df[self.split_col] == "test"],
    }

    # Patient-level overlap count across splits.
    if self.case_col in df.columns:
      split_sets = {}
      for split_name, split_df in split_frames.items():
        split_sets[split_name] = set(split_df[self.case_col].dropna().astype(str).tolist())
      overlap_cases = set()
      split_names = list(split_sets.keys())
      for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
          overlap_cases |= (split_sets[split_names[i]] & split_sets[split_names[j]])
      diag["case_overlap_count"] = int(len(overlap_cases))
    else:
      diag["case_overlap_count"] = None

    if self.center_col in df.columns:
      diag["center_distribution"] = {
        str(k): int(v)
        for k, v in df[self.center_col].value_counts(dropna=False).to_dict().items()
      }
    else:
      diag["center_distribution"] = {}

    class_min_per_split: Dict[str, Dict[str, int]] = {"primary": {}, "secondary": {}}
    for split_name, split_df in split_frames.items():
      if len(split_df) == 0:
        class_min_per_split["primary"][split_name] = 0
        class_min_per_split["secondary"][split_name] = 0
        continue
      primary_counts = split_df["label_name"].value_counts(dropna=True)
      class_min_per_split["primary"][split_name] = int(primary_counts.min()) if len(primary_counts) > 0 else 0

      if "label_secondary_name" in split_df.columns:
        secondary_counts = split_df["label_secondary_name"].value_counts(dropna=True)
        class_min_per_split["secondary"][split_name] = int(secondary_counts.min()) if len(secondary_counts) > 0 else 0
      else:
        class_min_per_split["secondary"][split_name] = 0
    diag["class_min_per_split"] = class_min_per_split

    secondary_mapping = self.secondary_label_mapping or {}
    if len(secondary_mapping) > 0 and "label_secondary_name" in df.columns:
      counts = df["label_secondary_name"].value_counts(dropna=True)
      zero_support = [name for name in secondary_mapping.keys() if int(counts.get(name, 0)) <= 0]
      diag["zero_support_registered_secondary"] = zero_support
    else:
      diag["zero_support_registered_secondary"] = []
    return diag

  def _dump_diag_file(self, diag: Dict[str, Any]) -> Optional[str]:
    if self.diag_output_file is None:
      return None
    output_path = Path(self.diag_output_file)
    os.makedirs(output_path.parent, exist_ok=True)
    output_path.write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path.as_posix()

  def _build_split_stat(self, df: pd.DataFrame, usable_df: pd.DataFrame, **split_counts):
    stat = dict(
      total=int(len(df)),
      usable=int(len(usable_df)),
      ignored=int(len(df) - len(usable_df)),
    )
    for split_name, split_count in split_counts.items():
      stat[split_name] = int(split_count)
    return stat

  def __call__(self):
    train_ratio, valid_ratio, test_ratio = self._validate_ratio()
    df, usable_df, _ = self._prepare_dataframe()
    stratify_values = self._resolve_stratify_column(usable_df)

    train_df, temp_df = train_test_split(
      usable_df,
      test_size=(1.0 - train_ratio),
      stratify=stratify_values,
      random_state=self.random_seed,
    )
    test_size_in_temp = test_ratio / (valid_ratio + test_ratio)
    stratify_temp = self._resolve_stratify_column(temp_df)
    valid_df, test_df = train_test_split(
      temp_df,
      test_size=test_size_in_temp,
      stratify=stratify_temp,
      random_state=self.random_seed,
    )

    df.loc[train_df.index, self.split_col] = "train"
    df.loc[valid_df.index, self.split_col] = "valid"
    df.loc[test_df.index, self.split_col] = "test"

    split_file = self._dump_split_file(df)
    split_diag = self._compute_split_diag(df, train_df=train_df, valid_df=valid_df, test_df=test_df)
    split_diag_file = self._dump_diag_file(split_diag)
    class_min_per_split = split_diag.get("class_min_per_split", {})
    for level_name, split_min in class_min_per_split.items():
      for split_name, min_count in split_min.items():
        if int(min_count) < self.min_class_samples:
          print_log(
            "Long-tail class detected: level={}, split={}, min_class_samples={} < threshold={}".format(
              level_name,
              split_name,
              min_count,
              self.min_class_samples,
            ),
            level="WARNING",
          )

    return dict(
      train=self._to_records(train_df),
      valid=self._to_records(valid_df),
      test=self._to_records(test_df),
      split_file=split_file,
      split_diag=split_diag,
      split_diag_file=split_diag_file,
      split_stat=self._build_split_stat(
        df,
        usable_df,
        train=len(train_df),
        valid=len(valid_df),
        test=len(test_df),
      ),
    )


@SOURCES.register_module()
class ExternalSheetSource(StratifiedSheetSource):
  def __init__(self, *args, center_name: str = "external", center_col: str = "center", **kwargs):
    super().__init__(*args, **kwargs)
    self.center_name = center_name
    self.center_col = center_col
    return

  def _normalize_external_inputs(self):
    if isinstance(self.sheet_file, dict):
      return [(str(k), v) for k, v in self.sheet_file.items()]
    if isinstance(self.sheet_file, (tuple, list)):
      items = []
      for index, item in enumerate(self.sheet_file):
        if isinstance(item, (tuple, list)) and len(item) == 2:
          items.append((str(item[0]), item[1]))
        else:
          items.append((f"{self.center_name}_{index}", item))
      return items
    return [(self.center_name, self.sheet_file)]

  def __call__(self):
    center_inputs = self._normalize_external_inputs()

    center_frames: List[pd.DataFrame] = []
    center_usable_frames: List[pd.DataFrame] = []
    split_stat_by_center = {}

    for center_name, sheet_file in center_inputs:
      sheet_df = self._read_sheet_from(sheet_file, self.sheet_name)
      df, usable_df, _ = self._prepare_dataframe(
        sheet_df,
        center_name=center_name,
        center_col=self.center_col,
      )
      df.loc[usable_df.index, self.split_col] = "test"
      center_frames.append(df)
      center_usable_frames.append(usable_df)
      split_stat_by_center[center_name] = self._build_split_stat(
        df,
        usable_df,
        test=len(usable_df),
      )

    if len(center_frames) == 0:
      full_df = pd.DataFrame(columns=["raw_file", "label_name", self.mapped_file_col, self.split_col])
      split_file = self._dump_split_file(full_df)
      split_diag = self._compute_split_diag(full_df)
      split_diag_file = self._dump_diag_file(split_diag)
      return dict(
        test=[],
        split_file=split_file,
        split_diag=split_diag,
        split_diag_file=split_diag_file,
        split_stat=dict(total=0, usable=0, ignored=0, test=0),
        split_stat_by_center=split_stat_by_center,
      )

    full_df = pd.concat(center_frames, ignore_index=True)
    usable_df = pd.concat(center_usable_frames, ignore_index=True)

    split_file = self._dump_split_file(full_df)
    split_diag = self._compute_split_diag(full_df, test_df=usable_df)
    split_diag_file = self._dump_diag_file(split_diag)

    return dict(
      test=self._to_records(usable_df),
      split_file=split_file,
      split_diag=split_diag,
      split_diag_file=split_diag_file,
      split_stat=self._build_split_stat(full_df, usable_df, test=len(usable_df)),
      split_stat_by_center=split_stat_by_center,
    )
