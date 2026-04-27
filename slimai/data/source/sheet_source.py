import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from slimai.helper.help_build import SOURCES


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
    split_col="split",
    output_split_file: Optional[str] = None,
    random_seed: int = 10482,
    split_ratio: Optional[Dict[str, float]] = None,
    label_mapping: Optional[Dict[str, int]] = None,
    path_mapping: Optional[Sequence[Tuple[str, str]]] = None,
    mapped_file_col: str = "EMBEDDING_MAPPED",
  ):
    self.sheet_file = sheet_file
    self.sheet_name = sheet_name
    self.file_col = file_col
    self.label_col = label_col
    self.split_col = split_col
    self.output_split_file = output_split_file
    self.random_seed = random_seed
    self.split_ratio = split_ratio or dict(train=0.8, valid=0.1, test=0.1)
    self.label_mapping = label_mapping
    self.path_mapping = list(path_mapping or [])
    self.mapped_file_col = mapped_file_col
    return

  def map_path(self, path):
    path = str(path)
    for src_prefix, dst_prefix in self.path_mapping:
      if path.startswith(src_prefix):
        return dst_prefix + path[len(src_prefix) :]
      path = path.replace(src_prefix, dst_prefix, 1)
    return path

  def _read_sheet(self):
    source = SheetSource(
      sheet_file=self.sheet_file,
      col_mapping={
        self.file_col: "raw_file",
        self.label_col: "label_name",
      },
      sheet_name=self.sheet_name,
      filter=None,
      apply=None,
    )
    data = source()
    df = pd.DataFrame(data)
    return df

  def _validate_ratio(self):
    train_ratio = self.split_ratio.get("train", 0.0)
    valid_ratio = self.split_ratio.get("valid", 0.0)
    test_ratio = self.split_ratio.get("test", 0.0)
    ratio_sum = train_ratio + valid_ratio + test_ratio
    assert abs(ratio_sum - 1.0) < 1e-8, (
      f"split_ratio must sum to 1.0, but got {self.split_ratio}"
    )
    return train_ratio, valid_ratio, test_ratio

  def __call__(self):
    train_ratio, valid_ratio, test_ratio = self._validate_ratio()
    df = self._read_sheet().copy()

    df["label_idx"] = df["label_name"].map(self.label_mapping) if self.label_mapping is not None else df["label_name"]
    df[self.mapped_file_col] = df["raw_file"].apply(lambda x: self.map_path(x) if pd.notna(x) else x)
    df[self.split_col] = "ignored"
    df["ignore_reason"] = ""

    valid_mask = (
      df["raw_file"].notna()
      & df["label_idx"].notna()
      & df[self.mapped_file_col].notna()
    )
    df.loc[df["raw_file"].isna(), "ignore_reason"] = "missing_embedding"
    df.loc[df["label_idx"].isna(), "ignore_reason"] = "unknown_label"
    df.loc[df[self.mapped_file_col].isna(), "ignore_reason"] = "invalid_mapped_path"

    usable_df = df[valid_mask].copy()

    train_df, temp_df = train_test_split(
      usable_df,
      test_size=(1.0 - train_ratio),
      stratify=usable_df["label_name"],
      random_state=self.random_seed,
    )
    test_size_in_temp = test_ratio / (valid_ratio + test_ratio)
    valid_df, test_df = train_test_split(
      temp_df,
      test_size=test_size_in_temp,
      stratify=temp_df["label_name"],
      random_state=self.random_seed,
    )

    df.loc[train_df.index, self.split_col] = "train"
    df.loc[valid_df.index, self.split_col] = "valid"
    df.loc[test_df.index, self.split_col] = "test"

    if self.output_split_file is not None:
      output_path = Path(self.output_split_file)
      os.makedirs(output_path.parent, exist_ok=True)
      output_sheet_name = f"{self.sheet_name}_split" if isinstance(self.sheet_name, str) else "split"
      df.to_excel(output_path, sheet_name=output_sheet_name, index=False)
      split_file = output_path.as_posix()
    else:
      split_file = None

    def to_records(_df):
      return list(zip(_df[self.mapped_file_col].tolist(), _df["label_name"].tolist()))

    return dict(
      train=to_records(train_df),
      valid=to_records(valid_df),
      test=to_records(test_df),
      split_file=split_file,
      split_stat=dict(
        total=int(len(df)),
        usable=int(len(usable_df)),
        ignored=int((~valid_mask).sum()),
        train=int(len(train_df)),
        valid=int(len(valid_df)),
        test=int(len(test_df)),
      ),
    )
