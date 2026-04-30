############################## 1. DATASET
import hashlib
from datetime import datetime

EXCEL_FILE = "/hzztai/slimai/_debug_/alias/projects/brexi/data/乳腺筛选结果_「复旦中山」全量.xlsx"
SHEET_NAME = "5-数据集"
OUTPUT_SPLIT_FILE = "/hzztai/slimai/_debug_/alias/projects/brexi/data/乳腺筛选结果_「复旦中山」全量_5-数据集_split.xlsx"

# Keep original Excel unchanged, and only map paths in runtime.
PATH_MAPPING = [
  ("/home/hzzt/shzs_embedding", "/.slimai/cache/shzs_embedding-valid"),
  # ("/home/hzzt/shzs_embedding", "/.slimai/cache/shzs_embedding"),
]

RANDOM_SEED = 10482
SPLIT_RATIO = dict(train=0.6, valid=0.2, test=0.2)
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

BREXI_SOURCE = dict(
  type="StratifiedSheetSource",
  sheet_file=EXCEL_FILE,
  sheet_name=SHEET_NAME,
  file_col="EMBEDDING",
  label_col="一级分类",
  split_col="split",
  output_split_file=OUTPUT_SPLIT_FILE,
  random_seed=RANDOM_SEED,
  split_ratio=SPLIT_RATIO,
  label_mapping=LABEL_MAPPING,
  path_mapping=PATH_MAPPING,
  mapped_file_col="EMBEDDING_MAPPED",
)

EXTERNAL_EXCEL_FILES = {
  # "external_a": "/hzztai/slimai/_debug_/alias/projects/brexi/data/乳腺筛选结果_「复旦中山」全量.xlsx",
  # "external_b": "/hzztai/slimai/_debug_/alias/projects/brexi/data/乳腺筛选结果_「复旦中山」全量.xlsx",
  # "external_c": "/hzztai/slimai/_debug_/alias/projects/brexi/data/乳腺筛选结果_「复旦中山」全量.xlsx",
}
EXTERNAL_OUTPUT_DIR = "/hzztai/slimai/_debug_/alias/projects/brexi/output"


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


def build_excel_dataset(split, *, balance=False, with_augment=False, desc=""):
  augmenter = TRAIN_AUGMENTER if with_augment else None
  return dict(
    type="TorchEmbeddingDataset",
    source=BREXI_SOURCE,
    split=split,
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

def build_external_source(center_name, sheet_file):
  return dict(
    type="ExternalSheetSource",
    sheet_file=sheet_file,
    sheet_name=SHEET_NAME,
    file_col="EMBEDDING",
    label_col="一级分类",
    split_col="split",
    output_split_file=f"{EXTERNAL_OUTPUT_DIR}/{center_name}_{SHEET_NAME}_external.xlsx",
    label_mapping=LABEL_MAPPING,
    path_mapping=PATH_MAPPING,
    mapped_file_col="EMBEDDING_MAPPED",
    center_name=center_name,
    center_col="center",
  )

def build_external_dataset(center_name, sheet_file, *, desc=""):
  return dict(
    type="TorchEmbeddingDataset",
    source=build_external_source(center_name, sheet_file),
    split="test",
    label_mapping=LABEL_MAPPING,
    balance=False,
    preload=False,
    embedding_tag="",
    augmenter=None,
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

def build_external_loader(center_name, sheet_file):
  return dict(
    dataset=build_external_dataset(
      center_name=center_name,
      sheet_file=sheet_file,
      desc=f"BREXI<{TARGET_NAME}:external:{center_name}>",
    ),
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    shuffle=False,
    pin_memory=True,
    collate_fn=dict(type="MILCollate"),
  )


batch_size = 4
num_workers = 4
persistent_workers = False

TRAIN_LOADER = dict(
  dataset=build_excel_dataset(
    "train",
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
    "valid",
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
    "test",
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

EXTERNAL_TEST_LOADERS = {}
if isinstance(EXTERNAL_EXCEL_FILES, dict) and len(EXTERNAL_EXCEL_FILES) > 0:
  EXTERNAL_TEST_LOADERS = {
    center_name: build_external_loader(center_name, sheet_file)
    for center_name, sheet_file in EXTERNAL_EXCEL_FILES.items()
  }
  EXTERNAL_TEST_LOADERS["external_pooled"] = build_external_loader(
    "external_pooled",
    EXTERNAL_EXCEL_FILES,
  )


############################## 2. MODEL
HEAD = dict(
  type="MLP",
  input_dim=EMBEDDING_DIM,
  output_dim=NUM_CLASSES,
  n_layer=3,
  dropout=0.5,
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
    dropout=0.5,
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
    lr=1e-4,
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
  f1=dict(
    type="torchmetrics.F1Score",
    task="multiclass",
    average="macro",
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

_PROJECT_ = "brexi"

_COMMENT_ = """
1. Excel-driven MIL training for brexi.
2. Runtime path mapping for EMBEDDING field via configurable PATH_MAPPING.
3. Stratified 80/10/10 split exported to output sheet.
4. TorchEmbeddingDataset reads NPU-compatible pickles via load_torch_pickle_compat.
"""
