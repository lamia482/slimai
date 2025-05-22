import pandas as pd
from pathlib import Path
from typing import Dict, List

from pyparsing import col
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
    
    df_dict = read_fn(self.sheet_file, sheet_name=sheet_name)
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
    return df.to_dict(orient="list")
