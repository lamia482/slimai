############################## 1. DATASET
########## 1.1 DATA TRANSFORM

flip_and_color_jitter = [
  # dict(type="RandomHorizontalFlip", p=0.5),
  # dict(type="RandomVerticalFlip", p=0.5), 
  # dict(type="RandomCrop", size=[1024, 1024]),
]

normalize = [
  dict(type="ToTensor"), 
  dict(type="Normalize", mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
]

train_view_transform = [
  dict(type="TorchTransform", transforms=[
    *flip_and_color_jitter, 
    *normalize, 
  ]), 
]

val_view_transform = [
  dict(type="TorchTransform", transforms=normalize), 
]

dataset_type = "SupervisedDataset"
train_dataset = dict(
  type=dataset_type, 
  dataset="/hzztai/projects/mtb/data/refine_20200422_lamia/train.pkl", 
  transform=train_view_transform, 
  desc="train mtb", 
  max_sample_num=None, 
  repeat=10,
  ann_keys=["instance"],
  filter_empty=True, 
  cache=False, 
)

val_dataset = dict(
  type=dataset_type, 
  dataset="/hzztai/projects/mtb/data/refine_20200422_lamia/train.pkl", 
  transform=val_view_transform, 
  desc="val mtb", 
  max_sample_num=None, 
  repeat=1,
  ann_keys=["instance"],
  filter_empty=True, 
  cache=False,
)

########## 1.3 DATA LOADER
batch_size = 2
num_workers = 1
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

############################## 2. MODEL
MODEL = dict(
  type="DetectionArch", 
  encoder=dict(
    backbone=dict(
      type="ViT",
      arch="nano",
      image_size=2000,
      patch_size=8,
      embed_dim=128, 
      drop_head=True,
      dropout=0.1, 
      attention_dropout=0.1, 
      cls_pooling=False,
    ), # [B, N, K]
    neck=dict(
      type="DETRQuery", 
      input_dim=128, 
      num_heads=4, 
      num_layers=3, 
      num_query=100, 
      dropout=0.1,
    ), # [B, Q, K]
  ), 
  decoder=dict(
    head=dict(
      type="DetectionHead",
      input_dim=128,
      num_classes=1,
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
    num_classes=1,
    eos_coef=0.1,
    cls_weight=1,
    box_weight=5,
    giou_weight=2,
  ), 
  solver=dict(
    type="torch.optim.AdamW",
    lr=1e-3,
    weight_decay=1e-2, 
    scheduler=dict(
      type="torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
      T_0=500,
      T_mult=1,
      eta_min=1e-4,
    ),
  ), 
)

############################## 3. RUNNER

RUNNER = dict(
  max_epoch=100, 
  compile=True, 
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
    iou_thresholds=[0.1],
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
del datetime, hashlib

_COMMENT_ = """
1. use no augmentation to train
"""