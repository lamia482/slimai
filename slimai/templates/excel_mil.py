############################## 1. DATASET
import hashlib
from datetime import datetime

EXCEL_FILE = "/hzztai/slimai/_debug_/alias/projects/brexi/data/乳腺筛选结果_复旦中山全量.xlsx"
SHEET_NAME = "5-数据集"
OUTPUT_SPLIT_FILE = "/hzztai/slimai/_debug_/alias/projects/brexi/output/乳腺筛选结果_复旦中山全量_5-数据集_split.xlsx"

# Keep original Excel unchanged, and only map paths in runtime.
PATH_MAPPING = [
  ("/home/hzzt/shzs_embedding", "/mnt/wangqiang/server/192.168.1.67/shzs_embedding"),
]

RANDOM_SEED = 10482
SPLIT_RATIO = dict(train=0.8, valid=0.1, test=0.1)
assert abs(sum(SPLIT_RATIO.values()) - 1.0) < 1e-8, "SPLIT_RATIO must sum to 1.0"

TARGET_NAME = "BREXI"
EMBEDDING_DIM = 1024  # K from FM embedding output, configurable for UNI/CONCH/... etc.
EMBEDDING_MAGNIFICATION = 20
EMBEDDING_KEY = "embedding"
COORDS_KEY = "x1_y1_dict"

LABEL_MAPPING = {
  "良性病变": 0,
  "良性肿瘤": 1,
  "癌前病变": 2,
  "恶性肿瘤（癌）": 3,
}
NUM_CLASSES = len(LABEL_MAPPING)

USE_DATA_BALANCE = False
USE_DATA_AUGMENTER = False
USE_FOCAL_LOSS = False


def map_embedding_path(path):
  path = str(path)
  for src_prefix, dst_prefix in PATH_MAPPING:
    if path.startswith(src_prefix):
      return dst_prefix + path[len(src_prefix) :]
    path = path.replace(src_prefix, dst_prefix, 1)
  return path


def _read_and_split_excel():
  pd = __import__("pandas")
  train_test_split = __import__(
    "sklearn.model_selection", fromlist=["train_test_split"]
  ).train_test_split
  os_module = __import__("os")
  Path = __import__("pathlib", fromlist=["Path"]).Path

  df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME).copy()
  df["label_name"] = df["一级分类"]
  df["label_idx"] = df["label_name"].map(LABEL_MAPPING)
  df["EMBEDDING_MAPPED"] = df["EMBEDDING"].apply(
    lambda x: map_embedding_path(x) if pd.notna(x) else x
  )
  df["split"] = "ignored"
  df["ignore_reason"] = ""

  valid_mask = (
    df["EMBEDDING"].notna()
    & df["label_idx"].notna()
    & df["EMBEDDING_MAPPED"].notna()
  )
  df.loc[df["EMBEDDING"].isna(), "ignore_reason"] = "missing_embedding"
  df.loc[df["label_idx"].isna(), "ignore_reason"] = "unknown_label"
  df.loc[df["EMBEDDING_MAPPED"].isna(), "ignore_reason"] = "invalid_mapped_path"

  usable_df = df[valid_mask].copy()
  train_df, temp_df = train_test_split(
    usable_df,
    test_size=(1.0 - SPLIT_RATIO["train"]),
    stratify=usable_df["label_name"],
    random_state=RANDOM_SEED,
  )
  test_size_in_temp = SPLIT_RATIO["test"] / (SPLIT_RATIO["valid"] + SPLIT_RATIO["test"])
  valid_df, test_df = train_test_split(
    temp_df,
    test_size=test_size_in_temp,
    stratify=temp_df["label_name"],
    random_state=RANDOM_SEED,
  )

  df.loc[train_df.index, "split"] = "train"
  df.loc[valid_df.index, "split"] = "valid"
  df.loc[test_df.index, "split"] = "test"

  output_path = Path(OUTPUT_SPLIT_FILE)
  os_module.makedirs(output_path.parent, exist_ok=True)
  df.to_excel(output_path, sheet_name=f"{SHEET_NAME}_split", index=False)

  def to_records(_df):
    return list(zip(_df["EMBEDDING_MAPPED"].tolist(), _df["label_name"].tolist()))

  return dict(
    train=to_records(train_df),
    valid=to_records(valid_df),
    test=to_records(test_df),
    split_file=output_path.as_posix(),
    split_stat=dict(
      total=int(len(df)),
      usable=int(len(usable_df)),
      ignored=int((~valid_mask).sum()),
      train=int(len(train_df)),
      valid=int(len(valid_df)),
      test=int(len(test_df)),
    ),
  )


SPLIT_DATA = _read_and_split_excel()


TRAIN_AUGMENTER = dict(
  type="EmbeddingAugmenter",
  transforms=[
    dict(type="RandomSelectFeatureBag", low=0.3, high=0.9, p=1.0),
    dict(type="RandomShuffleFeatureBag", p=0.5),
    dict(type="RandomFeatureDropout", p=0.1, q=0.1),
    dict(type="AddGaussianNoise", std=None, p=0.1),
    dict(type="MixupPatches", alpha=0.1, p=0.1),
  ],
)


def build_excel_dataset(records, *, balance=False, with_augment=False, desc=""):
  augmenter = TRAIN_AUGMENTER if with_augment else None
  return dict(
    type="TorchEmbeddingDataset",
    records=records,
    label_mapping=LABEL_MAPPING,
    balance=balance,
    preload=False,
    embedding_tag="",
    augmenter=augmenter,
    use_cache=True,
    cache_embedding_key="embedding",
    cache_visual_key="visual",
    max_sample_num=None,
    repeat=1,
    embedding_key=EMBEDDING_KEY,
    coords_key=COORDS_KEY,
    embedding_magnification=EMBEDDING_MAGNIFICATION,
    expected_embedding_dim=EMBEDDING_DIM,
    desc=desc,
  )


batch_size = 1
num_workers = 4
persistent_workers = False

TRAIN_LOADER = dict(
  dataset=build_excel_dataset(
    SPLIT_DATA["train"],
    balance=USE_DATA_BALANCE,
    with_augment=USE_DATA_AUGMENTER,
    desc=f"BREXI<{TARGET_NAME}:train>",
  ),
  batch_size=batch_size,
  num_workers=num_workers,
  persistent_workers=persistent_workers,
  shuffle=True,
  pin_memory=True,
  collate_fn=dict(type="MILCollate"),
)

VALID_LOADER = dict(
  dataset=build_excel_dataset(
    SPLIT_DATA["valid"],
    balance=False,
    with_augment=False,
    desc=f"BREXI<{TARGET_NAME}:valid>",
  ),
  batch_size=batch_size,
  num_workers=num_workers,
  persistent_workers=persistent_workers,
  shuffle=False,
  pin_memory=True,
  collate_fn=dict(type="MILCollate"),
)

TEST_LOADER = dict(
  dataset=build_excel_dataset(
    SPLIT_DATA["test"],
    balance=False,
    with_augment=False,
    desc=f"BREXI<{TARGET_NAME}:test>",
  ),
  batch_size=batch_size,
  num_workers=num_workers,
  persistent_workers=persistent_workers,
  shuffle=False,
  pin_memory=True,
  collate_fn=dict(type="MILCollate"),
)


############################## 2. MODEL
HEAD = dict(
  type="MLP",
  input_dim=EMBEDDING_DIM,
  output_dim=NUM_CLASSES,
  n_layer=1,
  dropout=0.1,
)


MODEL = dict(
  type="MIL",
  embedding_group_size=1,
  freeze_backbone=False,
  backbone=dict(type="torch.nn.Identity"),
  neck=dict(
    type="ABMIL",
    input_dim=EMBEDDING_DIM,
    hidden_dim=EMBEDDING_DIM,
    attention="gated",
    dropout=0.1,
  ),
  head=HEAD,
  loss=dict(
    type="MILLoss",
    atten_loss=False,
    cls_loss=(
      dict(type="SoftmaxFocalLoss", num_classes=NUM_CLASSES, alpha=None, gamma=2.0)
      if USE_FOCAL_LOSS
      else dict(type="torch.nn.CrossEntropyLoss", label_smoothing=0.0)
    ),
  ),
  solver=dict(
    type="torch.optim.AdamW",
    lr=1e-3,
    weight_decay=1e-2,
    scheduler=dict(
      type="torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
      T_0=500,
      T_mult=1,
      eta_min=1e-5,
    ),
  ),
)


############################## 3. RUNNER
RUNNER = dict(
  max_epoch=12,
  compile=False,
  checkpointing=False,
  gradient=dict(
    amp=True,
    clip=3.0,
    accumulation_every_n_steps=1,
  ),
  logger=dict(
    log_level="INFO",
    log_dir="logs",
    log_every_n_steps=1,
    log_precision=".8f",
    log_loss_precision=".8f",
    log_latency_precision=".3f",
  ),
  ckpt=dict(
    save_dir="ckpts",
    save_every_n_epochs=1,
    keep_max=5,
    keep_best=True,
    keep_latest=True,
    eval_every_n_epochs=1,
  ),
  resume=dict(
    enable=True,
    resume_from="latest",
    load_from=None,
  ),
)


############################## 4. METRIC
METRIC = dict(
  type="BasicClassificationMetric",
  acc=dict(
    type="torchmetrics.Accuracy",
    task="multiclass",
    num_classes=NUM_CLASSES,
    sync_on_compute=False,
  ),
  auc=dict(
    type="torchmetrics.AUROC",
    task="multiclass",
    num_classes=NUM_CLASSES,
    sync_on_compute=False,
  ),
  kappa=dict(
    type="torchmetrics.CohenKappa",
    task="multiclass",
    num_classes=NUM_CLASSES,
    sync_on_compute=False,
  ),
)


############################## ALL SET
signature = datetime.now().strftime(
  "%Y%m%d-{:s}".format(
    hashlib.md5(
      ("+".join(map(str, (TRAIN_LOADER, MODEL)))).encode(encoding="UTF-8")
    ).hexdigest()[:8]
  )
)


############################## CLEAR FOR DUMP
del datetime, hashlib

_PROJECT_ = "mil"

_COMMENT_ = """
1. Excel-driven MIL training for brexi.
2. Runtime path mapping for EMBEDDING field via configurable PATH_MAPPING.
3. Stratified 80/10/10 split exported to output sheet.
4. TorchEmbeddingDataset reads embedding payloads directly via torch.load.
"""
