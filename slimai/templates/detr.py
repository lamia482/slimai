############################## 1. DATASET
########## 1.1 DATA TRANSFORM

flip_and_color_jitter = [
  dict(type="RandomIoUCrop", min_scale=0.8, max_scale=1.2, min_aspect_ratio=0.5, max_aspect_ratio=2.0),
  dict(type="SanitizeBoundingBoxes", min_size=1, min_area=1),
  dict(type="RandomHorizontalFlip", p=0.5),
  dict(type="RandomVerticalFlip", p=0.5),
  dict(type="RandomChoice", transforms=[
    dict(type="RandomCrop", size=[1280, 1280]),
    dict(type="Resize", size=[1280, 1280]),
  ], p=1.0),
  dict(type="SanitizeBoundingBoxes", min_size=1, min_area=1),
]

import torch
normalize = [
  dict(type="ToDtype", dtype=torch.float32, scale=True), 
  dict(type="Normalize", mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
]

train_view_transform = [
  dict(type="TorchTransform", transforms=[
    *flip_and_color_jitter, 
    *normalize, 
  ]), 
]

val_view_transform = [
  dict(type="TorchTransform", transforms=[
    *normalize
  ]), 
]

class_names = ["ABNORMAL", "CC", "FUNGI", "ACTINO", "TRI"]
dataset_type = "SupervisedDataset"
train_dataset = dict(
  type=dataset_type, 
  dataset="/data/wangqiang/Dataset/HMU-OS-TCT/V1.0/train.pkl", 
  class_names=class_names,
  transform=train_view_transform, 
  desc="train tct", 
  max_sample_num=None, 
  repeat=1,
  ann_keys=["instance"],
  filter_empty=False, 
  cache=False, 
)

val_dataset = dict(
  type=dataset_type, 
  dataset="/data/wangqiang/Dataset/HMU-OS-TCT/V1.0/valid.pkl", 
  class_names=class_names,
  transform=val_view_transform, 
  desc="val tct", 
  max_sample_num=None, 
  repeat=1,
  ann_keys=["instance"],
  filter_empty=False, 
  cache=False,
)

test_dataset = dict(
  type=dataset_type, 
  dataset="/data/wangqiang/Dataset/HMU-OS-TCT/V1.0/test.pkl", 
  class_names=class_names,
  transform=val_view_transform, 
  desc="test tct", 
  max_sample_num=None, 
  repeat=1,
  ann_keys=["instance"],
  filter_empty=False, 
  cache=False,
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
  collate_fn=dict(type="DataCollate"),
)

VALID_LOADER = dict(
  dataset=val_dataset,
  batch_size=batch_size,
  num_workers=num_workers,
  persistent_workers=persistent_workers,
  shuffle=False,
  pin_memory=True,
  collate_fn=dict(type="DataCollate"),
)

TEST_LOADER = dict(
  dataset=test_dataset,
  batch_size=batch_size,
  num_workers=num_workers,
  persistent_workers=persistent_workers,
  shuffle=False,
  pin_memory=True,
  collate_fn=dict(type="DataCollate"),
)

############################## 2. MODEL
MODEL = dict(
  type="DetectionArch", 
  encoder=dict(
    backbone=dict(
      type="ViT",
      arch="small",
      image_size=224,
      patch_size=16,
      embed_dim=384, 
      drop_head=True,
      dropout=0.1, 
      attention_dropout=0.1, 
      cls_pooling=False,
    ), # [B, N, K]
    neck=dict(
      type="DETRQuery", 
      input_dim=384, 
      num_heads=4, 
      num_layers=3, 
      num_query=100, 
      dropout=0.1,
    ), # [B, Q, K]
  ), 
  decoder=dict(
    head=dict(
      type="DetectionHead",
      input_dim=384,
      num_classes=len(class_names),
      num_layers=3,
      dropout=0.1,
    ), # [B, Q, n_classes] + [B, Q, 4]
  ),
  loss=dict(
    type="DETRLoss",
    matcher=dict(
      cost_class=1,
      cost_bbox=1,
      cost_giou=1,
    ),
    use_focal_loss=True,
    num_classes=len(class_names),
    eos_coef=0.1,
    cls_weight=1,
    box_weight=5,
    giou_weight=2,
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
  max_epoch=100, 
  compile=False, 
  checkpointing=True, 

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
  ), 

  ckpt=dict(
    save_dir="ckpts",
    save_every_n_epochs=1, 
    keep_max=5,
    keep_best=True, # keep ckpt with minimum loss on VALID dataset
    keep_latest=True, # keep ckpt link to latest epoch
    eval_every_n_epochs=1, 
    n_vis_on_eval=8, # random render n images on eval
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
  type="BasicDetectionMetric",
  ap=dict(
    type="torchmetrics.detection.MeanAveragePrecision", 
    box_format="xyxy",
    iou_type="bbox",
    # iou_thresholds=[0.1],
    class_metrics=True,
    sync_on_compute=False,
  )
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
del datetime, hashlib, torch

_PROJECT_ = "detr"

_COMMENT_ = """

"""