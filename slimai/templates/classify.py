############################## 1. DATASET
########## 1.1 DATA TRANSFORM

flip_and_color_jitter = [
  dict(type="RandomResizedCrop", size=(32, 32), scale=(0.8, 1.2), ratio=(0.8, 1.2), antialias=True),
  dict(type="RandomHorizontalFlip", p=0.5),
  dict(type="RandomVerticalFlip", p=0.5),
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
    dict(type="Resize", size=(32, 32)),
    *normalize, 
  ]), 
]

class_names = list(map(str, range(10)))
dataset_type = "SupervisedDataset"
train_dataset = dict(
  type=dataset_type, 
  dataset=dict(
    type="TorchSource", 
    dataset="MNIST", 
    root="/.slimai/cache/dataset/mnist", 
    train=True, 
    download=True, 
  ), 
  class_names=class_names,
  transform=train_view_transform, 
  desc="train mnist", 
  max_sample_num=None, 
  repeat=1,
  ann_keys=["label"],
  filter_empty=False, 
  cache=False, 
)

val_dataset = dict(
  type=dataset_type, 
  dataset=dict(
    type="TorchSource", 
    dataset="MNIST", 
    root="/.slimai/cache/dataset/mnist", 
    train=False, 
    download=True, 
  ), 
  class_names=class_names,
  transform=val_view_transform, 
  desc="val mnist", 
  max_sample_num=None, 
  repeat=1,
  ann_keys=["label"],
  filter_empty=False, 
  cache=False,
)

########## 1.3 DATA LOADER
batch_size = 128
num_workers = 4
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
  type="ClassificationArch",
  backbone=dict(
    type="ViT",
    arch="small",
    patch_size=4,
    drop_head=True,
  ), 
  neck=None,
  head=dict(
    type="MLP",
    input_dim=384,
    hidden_dim=1024,
    bottleneck_dim=128, 
    output_dim=len(class_names),
    n_layer=1,
    act="gelu",
    norm=None,
    dropout=0.5,
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
  max_epoch=32, 
  compile=False, # accelerate training by compiling the model
  checkpointing=False, # slow down training by checkpointing the model

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
  type="BasicClassificationMetric",
  acc=dict(
    type="torchmetrics.Accuracy",
    task="multiclass",
    num_classes=len(class_names),
    sync_on_compute=False,
  ),
  auc=dict(
    type="torchmetrics.AUROC",
    task="multiclass",
    num_classes=len(class_names),
    sync_on_compute=False,
  ),
  kappa=dict(
    type="torchmetrics.CohenKappa",
    task="multiclass",
    num_classes=len(class_names),
    sync_on_compute=False,
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
del datetime, hashlib, torch

_PROJECT_ = "classify"

_EXPERIMENT_ = "mnist"

_COMMENT_ = """

"""