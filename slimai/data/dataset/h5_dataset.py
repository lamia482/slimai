import os.path as osp
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import mmengine
import torch
from tqdm import tqdm

from slimai.helper.help_build import DATASETS, build_source, build_transform
from slimai.helper.help_utils import print_log
from slimai.helper.utils.cache import get_cacher

from .mil_dataset import MILDataset
from .sample_strategy import SampleStrategy


LabelType = Union[int, str]
RecordLabelType = Union[LabelType, Dict[str, Any]]
RecordType = Tuple[str, RecordLabelType]


@DATASETS.register_module()
class H5Dataset(MILDataset):
  def __init__(
    self,
    pkl_file,
    key,
    *,
    label_mapping: Optional[Dict[LabelType, int]] = None,
    secondary_label_mapping: Optional[Dict[LabelType, int]] = None,
    balance: bool = False,
    sample_strategy: Optional[str] = None,
    preload: bool = False,
    augmenter=None,
    embedding_tag: str = ".kfb_feat",
    use_cache: bool = True,
    cache_embedding_key: str = "embedding",
    cache_visual_key: str = "visual",
    max_sample_num: Optional[int] = None,
    repeat: int = 1,
    desc: Optional[str] = None,
    **kwargs,
  ):
    # NOTE:
    # MILDataset -> SupervisedDataset already provides class balance via sample_strategy.
    # We avoid duplicating balance logic and use SampleStrategy when balance=True.
    self.pkl_file = pkl_file
    self.key = key
    self.desc = desc or f"H5Dataset<{key}>"
    self.label_mapping = self._normalize_label_mapping(label_mapping)
    self.embedding_tag = embedding_tag
    self.use_cache = use_cache
    self.cache_embedding_key = cache_embedding_key
    self.cache_visual_key = cache_visual_key
    self.max_sample_num = max_sample_num
    self.repeat = repeat
    self.cacher = get_cacher()
    self.embeddings = {}

    if augmenter is None:
      self.augmenter = None
    elif callable(augmenter):
      self.augmenter = augmenter
    else:
      self.augmenter = build_transform(augmenter)

    records = mmengine.load(self.pkl_file)["dataset"][self.key]
    valid_records: List[RecordType] = self.filter_invalid_records(records)

    self.files = [self.to_embedding_path(h5_path) for h5_path, _ in valid_records]
    self.ann_keys = ["label"]
    self.annotations = dict(
      label=[self.map_label(label) for _, label in valid_records],
    )
    self.indices = list(range(len(self.files)))
    if sample_strategy is None and balance:
      sample_strategy = "balance"
    if sample_strategy is not None:
      self.indices = SampleStrategy.update_indices(
        self.annotations,
        self.ann_keys,
        sample_strategy,
        self.indices,
      )
    self.class_names = self._infer_class_names()

    if preload:
      for embed_path in tqdm(self.files, desc=f"Preloading<{self.key}>"):
        self.get_embedding(embed_path)

    print_log(f"Dataset {self}", level="INFO")
    return

  def _normalize_label_mapping(self, label_mapping):
    if isinstance(label_mapping, str):
      label_mapping = eval(label_mapping)
    return label_mapping

  def _infer_class_names(self):
    if not self.annotations or not self.annotations.get("label"):
      return ["0"]
    labels = self.annotations["label"]
    max_label = max(labels)
    return [str(i) for i in range(int(max_label) + 1)]

  def _infer_secondary_class_names(self):
    if not self.annotations or not self.annotations.get("label_secondary"):
      return []
    labels = self.annotations["label_secondary"]
    max_label = max(labels)
    return [str(i) for i in range(int(max_label) + 1)]

  @property
  def length(self):
    size = len(self.indices)
    if self.max_sample_num is not None:
      size = min(size, self.max_sample_num)
    return size

  def __len__(self):
    return int(self.length * self.repeat)

  def __getitem__(self, item):
    item = item % self.length
    item = self.indices[item]
    return self.select_sample(item)

  def __str__(self):
    return (
      f"Total {len(self)} samples(selected from {len(self.files)} files, "
      f"max_sample_num={self.max_sample_num}, repeat={self.repeat})\n"
      f"\tDataset file: {self.pkl_file}\n"
      f"\tKey: {self.key}\n"
      f"\tEmbedding tag: {self.embedding_tag}\n"
      f"\tDescription: {self.desc}\n"
    )

  __repr__ = __str__

  def to_embedding_path(self, h5_path: str) -> str:
    if self.embedding_tag in ["", ".kfb_feat"]:
      return h5_path
    return h5_path.replace(".kfb_feat", self.embedding_tag)

  def map_label(self, label: LabelType) -> int:
    if self.label_mapping is None:
      return int(label)
    return int(self.label_mapping[label])

  def map_secondary_label(self, label: Optional[LabelType]) -> int:
    if label is None:
      return -1
    secondary_mapping = getattr(self, "secondary_label_mapping", None)
    if secondary_mapping is None:
      return int(label)
    return int(secondary_mapping[label])

  def _normalize_record(self, record: RecordType):
    embed_path, labels = record
    if isinstance(labels, dict):
      label = labels.get("label", None)
      label_secondary = labels.get("label_secondary", None)
      label_secondary_local = labels.get("label_secondary_local", None)
      if label_secondary_local is not None:
        label_secondary_local = int(label_secondary_local)
      return embed_path, label, label_secondary, label_secondary_local
    return embed_path, labels, None, None

  def filter_invalid_records(self, records: List[RecordType]) -> List[RecordType]:
    valid_records = []
    for record in records:
      h5_path, label, label_secondary, label_secondary_local = self._normalize_record(record)
      embed_path = self.to_embedding_path(h5_path)
      if not osp.exists(embed_path):
        print_log(f"Embedding file not found: {embed_path}. Skip this record.", level="WARNING")
        continue
      mapped_label = self.map_label(label)
      if mapped_label < 0:
        print_log(f"Label is negative for {embed_path}. Skip this record.", level="WARNING")
        continue
      if label_secondary is not None:
        mapped_secondary = self.map_secondary_label(label_secondary)
        if mapped_secondary < 0:
          print_log(f"Secondary label is negative for {embed_path}. Skip this record.", level="WARNING")
          continue
      if label_secondary is not None and label_secondary_local is not None and int(label_secondary_local) < 0:
        print_log(f"Secondary local label is negative for {embed_path}. Skip this record.", level="WARNING")
        continue
      valid_records.append(record)
    return valid_records

  def get_embedding(self, h5_path: str):
    if h5_path in self.embeddings:
      return self.embeddings[h5_path]
    with h5py.File(h5_path, "r") as fp:
      embeddings = fp["features"][:]  # type: ignore
      coords = fp["coords"][:]  # type: ignore
    embeddings = torch.from_numpy(embeddings).float()
    coords = torch.from_numpy(coords).float()
    self.embeddings[h5_path] = (embeddings, coords)
    return self.embeddings[h5_path]

  def apply_augmenter(self, embeddings: torch.Tensor) -> torch.Tensor:
    if self.augmenter is None:
      return embeddings
    if hasattr(self.augmenter, "transform"):
      return self.augmenter.transform(embeddings)
    try:
      data = dict(meta=dict(embedding=embeddings))
      data = self.augmenter(data)
      return data["meta"]["embedding"]
    except Exception:
      return self.augmenter(embeddings)

  def select_sample(self, item):
    embed_path = self.files[item]
    label = self.annotations["label"][item]
    label_secondary = (
      self.annotations["label_secondary"][item]
      if "label_secondary" in self.annotations
      else None
    )
    label_secondary_local = (
      self.annotations["label_secondary_local"][item]
      if "label_secondary_local" in self.annotations
      else None
    )
    embedding, coords = self.get_embedding(embed_path)
    embedding = self.apply_augmenter(embedding)

    embedding_cache_key = "+".join(
      map(
        str,
        [
          "dataset",
          self.__class__.__name__,
          embed_path,
          self.cache_embedding_key,
        ],
      )
    )
    visual_cache_key = "+".join(
      map(
        str,
        [
          "dataset",
          self.__class__.__name__,
          embed_path,
          self.cache_visual_key,
        ],
      )
    )

    data = dict(
      indice=item,
      image=None,
      label=label,
      latency=dict(
        data_select_latency=0.0,
        data_loader_latency=0.0,
        data_to_pil_latency=0.0,
        data_wrap_latency=0.0,
        data_transform_latency=0.0,
      ),
      meta=dict(
        wsi_shrink=None,
        patch_num=len(embedding),
        use_cache=self.use_cache,
        visual_file=visual_cache_key,
        embedding=embedding,
        embedding_key=embedding_cache_key,
        visual_key=visual_cache_key,
        coords=coords,
        h5_path=embed_path,
      ),
    )
    if label_secondary is not None:
      data["label_secondary"] = int(label_secondary)
    if label_secondary_local is not None:
      data["label_secondary_local"] = int(label_secondary_local)
    return data


@DATASETS.register_module()
class TorchEmbeddingDataset(H5Dataset):
  def __init__(
    self,
    records=None,
    *,
    label_mapping: Optional[Dict[LabelType, int]] = None,
    secondary_label_mapping: Optional[Dict[LabelType, int]] = None,
    balance: bool = False,
    sample_strategy: Optional[str] = None,
    preload: bool = False,
    augmenter=None,
    embedding_tag: str = "",
    use_cache: bool = True,
    cache_embedding_key: str = "embedding",
    cache_visual_key: str = "visual",
    max_sample_num: Optional[int] = None,
    repeat: int = 1,
    desc: Optional[str] = None,
    embedding_key: str = "embedding",
    coords_key: str = "x1_y1_dict",
    h5_embedding_key: str = "features",
    h5_coords_key: str = "coords",
    h5_schema_version: Optional[str] = None,
    embedding_magnification: Optional[Union[int, str]] = None,
    expected_embedding_dim: Optional[int] = None,
    source=None,
    split: Optional[str] = None,
    **kwargs,
  ):
    self.pkl_file = "<in-memory-records>"
    self.key = kwargs.pop("key", "excel")
    self.desc = desc or "TorchEmbeddingDataset<excel>"
    self.label_mapping = self._normalize_label_mapping(label_mapping)
    self.secondary_label_mapping = self._normalize_label_mapping(secondary_label_mapping)
    self.embedding_tag = embedding_tag
    self.use_cache = use_cache
    self.cache_embedding_key = cache_embedding_key
    self.cache_visual_key = cache_visual_key
    self.max_sample_num = max_sample_num
    self.repeat = repeat
    self.cacher = get_cacher()
    self.embeddings = {}

    self.embedding_key = embedding_key
    self.coords_key = coords_key
    self.h5_embedding_key = h5_embedding_key
    self.h5_coords_key = h5_coords_key
    self.h5_schema_version = h5_schema_version
    self.embedding_magnification = embedding_magnification
    self.expected_embedding_dim = expected_embedding_dim
    self.split = split
    self.source = source
    self.split_file = None
    self.split_stat = None
    self.split_diag = None
    self.split_diag_file = None

    if augmenter is None:
      self.augmenter = None
    elif callable(augmenter):
      self.augmenter = augmenter
    else:
      self.augmenter = build_transform(augmenter)

    if records is None:
      if source is None:
        raise ValueError("Either records or source must be provided for TorchEmbeddingDataset.")
      source_fn = build_source(source)
      source_data = source_fn()
      if split is None:
        raise ValueError("split must be provided when source is used.")
      if split not in source_data:
        raise KeyError(f"Split '{split}' not found in source output. Available keys: {list(source_data.keys())}")
      records = source_data[split]
      self.split_file = source_data.get("split_file", None)
      self.split_stat = source_data.get("split_stat", None)
      self.split_diag = source_data.get("split_diag", None)
      self.split_diag_file = source_data.get("split_diag_file", None)

    valid_records: List[RecordType] = self.filter_invalid_records(records)
    normalized_records = [self._normalize_record(record) for record in valid_records]
    self.files = [self.to_embedding_path(embed_path) for embed_path, _, _, _ in normalized_records]

    labels = [self.map_label(label) for _, label, _, _ in normalized_records]
    has_secondary_labels = any(label_secondary is not None for _, _, label_secondary, _ in normalized_records)
    self.ann_keys = ["label"]
    self.annotations = dict(label=labels)
    if has_secondary_labels:
      self.ann_keys.extend(["label_secondary", "label_secondary_local"])
      self.annotations["label_secondary"] = [
        self.map_secondary_label(label_secondary)
        for _, _, label_secondary, _ in normalized_records
      ]
      self.annotations["label_secondary_local"] = [
        int(-1 if label_secondary_local is None else label_secondary_local)
        for _, _, _, label_secondary_local in normalized_records
      ]
    self.indices = list(range(len(self.files)))
    if sample_strategy is None and balance:
      sample_strategy = "balance"
    if sample_strategy is not None:
      self.indices = SampleStrategy.update_indices(
        self.annotations,
        self.ann_keys,
        sample_strategy,
        self.indices,
      )
    self.class_names = self._infer_class_names()
    self.class_names_secondary = self._infer_secondary_class_names()

    if preload:
      for embed_path in tqdm(self.files, desc=f"Preloading<{self.key}>"):
        self.get_embedding(embed_path)

    print_log(f"Dataset {self}", level="INFO")
    return

  def __str__(self):
    return (
      f"Total {len(self)} samples(selected from {len(self.files)} files, "
      f"max_sample_num={self.max_sample_num}, repeat={self.repeat})\n"
      f"\tDataset file: {self.pkl_file}\n"
      f"\tKey: {self.key}\n"
      f"\tEmbedding tag: {self.embedding_tag}\n"
      f"\tEmbedding key: {self.embedding_key}\n"
      f"\tCoordinates key: {self.coords_key}\n"
      f"\tH5 embedding key: {self.h5_embedding_key}\n"
      f"\tH5 coordinates key: {self.h5_coords_key}\n"
      f"\tH5 schema version: {self.h5_schema_version}\n"
      f"\tEmbedding magnification: {self.embedding_magnification}\n"
      f"\tExpected embedding dim: {self.expected_embedding_dim}\n"
      f"\tSplit: {self.split}\n"
      f"\tSplit file: {self.split_file}\n"
      f"\tSplit stat: {self.split_stat}\n"
      f"\tSplit diag file: {self.split_diag_file}\n"
      f"\tDescription: {self.desc}\n"
    )

  __repr__ = __str__

  def _select_group_data(self, data, data_name):
    if not isinstance(data, dict):
      return data

    if len(data) == 0:
      raise ValueError(f"{data_name} dict is empty.")

    if self.embedding_magnification is None:
      key = list(data.keys())[0]
    elif self.embedding_magnification in data:
      key = self.embedding_magnification
    elif str(self.embedding_magnification) in data:
      key = str(self.embedding_magnification)
    else:
      raise KeyError(
        f"{data_name} magnification {self.embedding_magnification} not found. Available keys: {list(data.keys())[:20]}"
      )
    return data[key]

  def _extract_embedding_and_coords(self, payload, embed_path: str):
    if not isinstance(payload, dict):
      raise ValueError(f"Embedding payload must be dict, but got {type(payload)} from {embed_path}.")

    if self.embedding_key not in payload:
      raise KeyError(
        f"Embedding key '{self.embedding_key}' not found in {embed_path}. "
        f"Available keys: {list(payload.keys())[:20]}"
      )

    embedding = self._select_group_data(payload[self.embedding_key], "embedding")
    coords = payload.get(self.coords_key, None)
    if coords is not None:
      coords = self._select_group_data(coords, "coords")
    return embedding, coords

  def _extract_embedding_and_coords_from_h5(self, embed_path: str):
    with h5py.File(embed_path, "r") as fp:
      if self.h5_schema_version is not None:
        schema_version = str(fp.attrs.get("schema_version", ""))
        if schema_version != str(self.h5_schema_version):
          raise ValueError(
            f"H5 schema mismatch in {embed_path}: got '{schema_version}', expected '{self.h5_schema_version}'."
          )

      if self.h5_embedding_key not in fp:
        raise KeyError(
          f"H5 embedding key '{self.h5_embedding_key}' not found in {embed_path}. "
          f"Available keys: {list(fp.keys())[:20]}"
        )
      embedding = fp[self.h5_embedding_key][:]

      coords = None
      if self.h5_coords_key in fp:
        coords = fp[self.h5_coords_key][:]
    return embedding, coords

  def _normalize_coords(self, coords, embedding: torch.Tensor, embed_path: str) -> torch.Tensor:
    if coords is None:
      return torch.zeros((embedding.shape[0], 2), dtype=torch.float32)

    coords = torch.as_tensor(coords).float()
    if coords.dim() == 1:
      coords = coords.unsqueeze(-1)

    if coords.shape[0] != embedding.shape[0]:
      print_log(
        f"Coordinates number mismatch in {embed_path}: coords={coords.shape[0]} vs embedding={embedding.shape[0]}. "
        "Fallback to zeros coordinates.",
        level="WARNING",
      )
      coords = torch.zeros((embedding.shape[0], max(2, coords.shape[-1])), dtype=torch.float32)
    return coords

  def get_embedding(self, embed_path: str):
    if embed_path in self.embeddings:
      return self.embeddings[embed_path]

    if str(embed_path).lower().endswith((".h5", ".hdf5")):
      embedding, coords = self._extract_embedding_and_coords_from_h5(embed_path)
    else:
      payload = torch.load(
        embed_path,
        map_location="cpu",
        weights_only=False,
      )
      embedding, coords = self._extract_embedding_and_coords(payload, embed_path)
    embedding = torch.as_tensor(embedding).float()
    if embedding.dim() != 2:
      raise ValueError(f"Embedding tensor must be 2D [N, K], got {tuple(embedding.shape)} from {embed_path}.")

    if self.expected_embedding_dim is not None and embedding.shape[-1] != self.expected_embedding_dim:
      raise ValueError(
        "Embedding dim mismatch in {}: got {}, expected {}. "
        "Please check EMBEDDING_DIM in config.".format(
          embed_path, embedding.shape[-1], self.expected_embedding_dim
        )
      )

    coords = self._normalize_coords(coords, embedding, embed_path)
    self.embeddings[embed_path] = (embedding, coords)
    return self.embeddings[embed_path]
