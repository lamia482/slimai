############################## 1. DATASET

DATASET_FILE = "/hzztai/projects/thca/cell_det/jupyter/dataset.pkl"
TARGET_NAME = "TBS" # TBS, BRAF
EMBEDDING_TAG = ".kfb_feat_UNI_GRAY"
EMBEDDING_DIM = 1024

USE_LABEL_MAPPING = "3C"
LABEL_MAPPING = {
  "2C": {1: 0, 2: 0, 3: -1, 4: -1, 5: 1, 6: 1},
  "3C": {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2},
  "4C": {1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 3},
}[USE_LABEL_MAPPING]
NUM_CLASSES = max(LABEL_MAPPING.values()) + 1
CLASS_NAMES = [str(i) for i in range(NUM_CLASSES)]

USE_DATA_BALANCE = False
USE_DATA_AUGMENTER = False
USE_FOCAL_LOSS = False
USE_BAG_HAT = False

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

if NUM_CLASSES == 3:
  HEAD = dict(
    type="THCAHeadC3",
    input_dim=EMBEDDING_DIM + int(USE_BAG_HAT),
    output_dim=NUM_CLASSES,
    hidden_dim=EMBEDDING_DIM,
    bottleneck_dim=max(64, EMBEDDING_DIM // 2),
    n_layer=1,
    act="gelu",
    norm="layer_norm",
    dropout=0.5,
  )
else:
  HEAD = dict(
    type="MLP",
    input_dim=EMBEDDING_DIM + int(USE_BAG_HAT),
    output_dim=NUM_CLASSES,
    hidden_dim=EMBEDDING_DIM,
    bottleneck_dim=max(64, EMBEDDING_DIM // 2),
    n_layer=1,
    act="gelu",
    norm="layer_norm",
    dropout=0.5,
  )


def build_h5_dataset(key, *, shuffle=False, balance=False, with_augment=False):
  augmenter = TRAIN_AUGMENTER if with_augment else None
  return dict(
    type="H5Dataset",
    pkl_file=DATASET_FILE,
    key=key,
    label_mapping=LABEL_MAPPING,
    balance=balance,
    preload=False,
    embedding_tag=EMBEDDING_TAG,
    augmenter=augmenter,
    use_cache=True,
    repeat=1,
    max_sample_num=None,
    desc=f"MIL<{key}>",
  )


batch_size = 32
num_workers = 8
persistent_workers = False

TRAIN_LOADER = dict(
  dataset=build_h5_dataset(
    f"DEV-K1-TRAIN-{TARGET_NAME}",
    shuffle=True,
    balance=USE_DATA_BALANCE,
    with_augment=USE_DATA_AUGMENTER,
  ),
  batch_size=batch_size,
  num_workers=num_workers,
  persistent_workers=persistent_workers,
  shuffle=True,
  pin_memory=True,
  collate_fn=dict(type="MILCollate"),
)

VALID_LOADER = dict(
  dataset=build_h5_dataset(f"DEV-K1-VAL-{TARGET_NAME}", shuffle=False),
  batch_size=batch_size,
  num_workers=num_workers,
  persistent_workers=persistent_workers,
  shuffle=False,
  pin_memory=True,
  collate_fn=dict(type="MILCollate"),
)

TEST_LOADER = dict(
  dataset=build_h5_dataset(f"ET1-TEST-{TARGET_NAME}", shuffle=False),
  batch_size=batch_size,
  num_workers=num_workers,
  persistent_workers=persistent_workers,
  shuffle=False,
  pin_memory=True,
  collate_fn=dict(type="MILCollate"),
)


############################## 2. MODEL
MODEL = dict(
  type="MIL",
  embedding_group_size=1,
  freeze_backbone=False,
  backbone=dict(type="torch.nn.Identity"),
  neck=dict(
    type="WMIL",
    input_dim=EMBEDDING_DIM,
    hidden_dim=EMBEDDING_DIM,
    attention="gated",
    dropout=0.5,
    keep_size_hat=USE_BAG_HAT,
    attention_temperature=1.0,
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
from datetime import datetime
import hashlib

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
1. MIL training with H5 precomputed embeddings.
2. Runner + MIL architecture with Identity backbone.
3. Optional embedding-level augmentation and focal loss.
"""
