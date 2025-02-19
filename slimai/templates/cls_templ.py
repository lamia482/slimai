############################## 1. DATASET
########## 1.1 DATA TRANSFORM
import cv2
train_transform = [
  dict(type="AlbuTransform", transforms=[
    dict(type="OneOf", transforms=[
      dict(type="LongestMaxSize", max_size=256),
      dict(type="Resize", height=256, width=256),
    ], p=0.8),
    dict(type="PadIfNeeded", min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255)),
    dict(type="VerticalFlip", p=0.5),
    dict(type="HorizontalFlip", p=0.5),
    dict(type="OneOf", transforms=[
      dict(type="Blur", blur_limit=3, p=0.5),
      dict(type="MotionBlur", blur_limit=3, p=0.5),
    dict(type="AdvancedBlur", blur_limit=(3, 5), sigma_x_limit=(0.2, 0.5), sigma_y_limit=(0.2, 0.5), rotate_limit=30, p=0.5),
      dict(type="Downscale", scale_min=0.75, scale_max=0.9, p=0.5),
      dict(type="ImageCompression", quality_lower=60, quality_upper=100, p=0.5),
    ], p=0.15),
    dict(type="Resize", height=448, width=448),
    dict(type="RandomCrop", height=384, width=384),
    dict(type="Normalize", mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    dict(type="ToTensorV2"),
  ])
]

test_transform = [
  dict(type="AlbuTransform", transforms=[
    dict(type="LongestMaxSize", max_size=256),
    dict(type="PadIfNeeded", min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255)),
    dict(type="CenterCrop", height=224, width=224),
    dict(type="Normalize", mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    dict(type="ToTensorV2"),
  ])
]

########## 1.2 DATA SET
CLASS_NAMES = (
  "ASC-US", "LSIL", "ASC-H", "HSIL", "AGC-N", 
  "TRI", "FUNGI", "WS", "CC", "ACTINO", 
  "GEC", "NILM", "Glycogen", "Repair", 
  "Debris"
)

from functools import partial
def std_func(dataset, phase):
  import mmengine
  import os.path as osp
  if isinstance(dataset, str):
    dataset = mmengine.load(dataset)
  dataset = dataset[f"{phase}_files"]
  dataset = {
    "version": "fake-version",
    "signature": "fake-signature",
    "class_names": CLASS_NAMES,
    "files": list(map(lambda file: file.replace(
        "/Volumes/ai/TCT/DataSet", 
        "/mnt/wangqiang/server/10.168.100.21/ai/internal/projects/hzztai/projects/tct/cell_det/data"), 
        dataset)),
    "annotations": dict(
      label=list(map(lambda file: CLASS_NAMES.index(osp.basename(osp.dirname(file))), dataset))
    )
  }
  return dataset

dataset_type = "SupervisedDataset"
train_dataset = dict(
  type=dataset_type, 
  dataset="/mnt/wangqiang/server/10.168.100.21/ai/internal/projects/hzztai/projects/research/aigc/jupyter/pathology_fms/dataset_split.pkl",
  std_func=partial(std_func, phase="train"), 
  ann_keys=["label"],
  transform=train_transform, 
  desc="molagu-train", 
  max_sample_num=None,
)
valid_dataset = dict(
  type=dataset_type, 
  dataset="/mnt/wangqiang/server/10.168.100.21/ai/internal/projects/hzztai/projects/research/aigc/jupyter/pathology_fms/dataset_split.pkl",
  std_func=partial(std_func, phase="valid"), 
  ann_keys=["label"],
  transform=test_transform, 
  desc="molagu-valid",
  max_sample_num=None,
)
test_dataset = dict(
  type=dataset_type, 
  dataset="/mnt/wangqiang/server/10.168.100.21/ai/internal/projects/hzztai/projects/research/aigc/jupyter/pathology_fms/dataset_split.pkl",
  std_func=partial(std_func, phase="test"), 
  ann_keys=["label"],
  transform=test_transform, 
  desc="molagu-test",
  max_sample_num=None,
)

########## 1.3 DATA LOADER
batch_size = 32
num_workers = 8
persistent_workers = True if num_workers > 0 else False
TRAIN_LOADER = dict(
  dataset=train_dataset,
  batch_size=batch_size,
  num_workers=num_workers,
  persistent_workers=persistent_workers,
  shuffle=True,
  pin_memory=True,
)
VALID_LOADER = dict(
  dataset=valid_dataset,
  batch_size=batch_size,
  num_workers=num_workers,
  persistent_workers=persistent_workers,
  shuffle=False,
)
TEST_LOADER = dict(
  dataset=test_dataset,
  batch_size=batch_size,
  num_workers=num_workers,
  persistent_workers=persistent_workers,
  shuffle=False,
)

############################## 2. MODEL
MODEL = dict(
  type="ClassificationArch",
  encoder=dict(
    backbone=dict(
      type="ViT",
      arch="base",
      patch_size=16,
      drop_head=True,
    ),
    neck=None,
  ), 
  decoder=dict(
    head=dict(
      type="MLP",
      input_dim=768,
      hidden_dim=2048,
      bottleneck_dim=256, 
      output_dim=65536,
      n_layer=1,
      act="gelu",
      norm=None,
      dropout=0.5,
    ),
  ),
  loss=dict(
    type="BasicClassificationLoss",
    cls_loss=dict(
      type="torch.nn.CrossEntropyLoss",
      label_smoothing=0.1,
    )
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

  gradient=dict(
    amp=True, 
    accumulation_every_n_steps=1,
    clip=None, 
    checkpointing=True, 
  ), 

  logger=dict(
    log_level="INFO",
    log_dir="logs",
    log_every_n_steps=1,
  ), 

  ckpt=dict(
    save_dir="ckpts",
    save_every_n_epochs=1,
    keep_max=3,
    keep_best=True, # keep ckpt with minimum loss on VALID dataset
    keep_latest=True, # keep ckpt link to latest epoch
  ),

  resume=dict(
    # resume from epoch > load_from
    enable=True,
    resume_from="latest", # best or latest, or epoch number, or ckpt path
    load_from=None,
  )
)


############################## 4. Metric
METRIC = dict(
  type="BasicClassificationMetric",
  acc=dict(
    type="torchmetrics.Accuracy",
    task="multiclass",
    num_classes=len(CLASS_NAMES)
    ), 
  kappa=dict(
    type="torchmetrics.CohenKappa",
    task="multiclass",
    num_classes=len(CLASS_NAMES),
  ),
)


############################## ALL SET
# create model signature by default like 20250208-50adae7c
# set static and specific signature for future auto-resume
from datetime import datetime
import hashlib
signature = datetime.now().strftime("%Y%m%d-{:s}".format(
  hashlib.md5(("+".join(map(str, (
    TRAIN_LOADER, MODEL
    )))).encode(encoding="UTF-8")
  ).hexdigest()[:8]
))

############################## CLEAR FOR DUMP
del datetime, hashlib