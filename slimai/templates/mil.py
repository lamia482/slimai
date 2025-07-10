############################## 1. DATASET
########## 1.1 DATA TRANSFORM

flip_color_jitter = [
  dict(type="RandomApply", transforms=[
    dict(type="RandomHorizontalFlip", p=0.5),
    dict(type="RandomVerticalFlip", p=0.5),
  ], p=1/30), # patch num ~= 18000
  dict(type="RandomApply", transforms=[
    dict(type="ColorJitter", brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1), 
  ], p=1/30), 
  dict(type="RandomApply", transforms=[
    dict(type="GaussianBlur", kernel_size=(5, 5), sigma=(0.5, 2.0)), 
  ], p=1/30), 
]

import torch
normalize = [
  dict(type="ToDtype", dtype=torch.float32, scale=True), 
  dict(type="Normalize", mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
]

loader = dict(
  type="RegionTileLoader", 
  magnification=10, 
  region=dict(xmin=0.15, ymin=0.15, xmax=0.85, ymax=0.85),  # -1 for full size
  cache=True, 
  num_threads=4,
)

train_view_transform = [
  dict(type="MILTransform", 
    shrink="tissue", 
    tile_size=256, 
    tile_stride=224, 
    random_crop_patch_size=224, # random crop n patch from each tile
    random_crop_patch_num=1, 
    transform_schema="individual",
    topk=0,
    shuffle=True, # shuffle patches
    padding_value=255,
    transforms=dict(
      type="TorchTransform", transforms=[*flip_color_jitter, *normalize],
    ),
  )]

val_view_transform = [
  dict(type="MILTransform", 
    shrink="tissue", 
    tile_size=224, 
    tile_stride=224, 
    random_crop_patch_size=None, # random crop n patch from each tile
    random_crop_patch_num=None, 
    transform_schema="individual",
    topk=0,
    shuffle=False,
    padding_value=255,
    transforms=dict(
      type="TorchTransform", transforms=normalize, 
    ), 
  )]

dataset_type = "MILDataset"
class_names = ["NILM", "ASC-US", "LSIL", "ASC-H", "HSIL", "AGC"]
basic_dataset = dict(
  type=dataset_type, 
  dataset=dict(
    type="SheetSource",
    sheet_file="/mnt/wangqiang/server/10.168.100.21/ai/yujie/切片分类库数据集/切片分类v0.4数据集/切片分类v0.4数据集.xlsx", 
    col_mapping={
      "wsi_path": "files",
      "label": "label",
      "data_set": "phase", 
      "制片方式": "producer", 
    },
    sheet_name=None,
    filter=[
      ("phase", "==", "train"), 
      ("producer", "==", "kfbio"), # ["bd", "kfbio", "thinprep"]
    ], 
    apply=[
      ("files", "lambda file: file.replace('/root/workspace/server21/AI/ai_21/yujie', '/mnt/wangqiang/server/10.168.100.21/ai/yujie')"), 
      ("label", f"lambda label: {class_names}.index(label)"), 
    ]
  ),
  sample_strategy=None,
  class_names=class_names, 
  std_func="lambda dataset: dict(files=dataset.pop('files'), annotations=dict(label=dataset.pop('label'), producer=dataset.pop('producer')))",
  transform=train_view_transform, 
  loader=loader, 
  desc="basic tct wsi mil", 
  max_sample_num=None, 
  repeat=1,
  ann_keys=["label"],
  cache=False, 
)

import copy
train_dataset = copy.deepcopy(basic_dataset)
train_dataset["dataset"]["filter"][0] = ("phase", "==", "train")
train_dataset["sample_strategy"] = "balance"
train_dataset["desc"] = "train tct wsi mil"

val_dataset = copy.deepcopy(basic_dataset) 
val_dataset["dataset"]["filter"][0] = ("phase", "==", "test")
val_dataset["desc"] = "val tct wsi mil"
val_dataset["transform"] = val_view_transform
val_dataset["cache_embedding"] = True
val_dataset["feature_extractor"] = dict(
  type="Plugin",
  module="/mnt/wangqiang/server/10.168.100.21/ai/nengwp/TCT/dino/infer_wq.py:get_backbone",
  weight="/mnt/wangqiang/server/10.168.100.21/ai/nengwp/TCT/dino/pretrain/dinov2_exp3/training_742052/teacher_checkpoint.pth" 
)

########## 1.3 DATA LOADER
batch_size = 1
num_workers = 1
prefetch_factor = 1
persistent_workers = False # True if num_workers > 0 else False

TRAIN_LOADER = dict(
  dataset=train_dataset,
  batch_size=batch_size,
  num_workers=num_workers,
  prefetch_factor=prefetch_factor,
  persistent_workers=persistent_workers,
  shuffle=True,
  pin_memory=True, 
  collate_fn=dict(
    type="MILCollate",
  )
)

VALID_LOADER = dict(
  dataset=val_dataset,
  batch_size=batch_size,
  num_workers=num_workers,
  prefetch_factor=prefetch_factor,
  persistent_workers=persistent_workers,
  shuffle=False,
  pin_memory=True,
  collate_fn=dict(
    type="MILCollate",
  )
)

############################## 2. MODEL
MODEL = dict(
  type="MIL", 
  embedding_group_size=16, 
  freeze_backbone=True,
  encoder=dict(
    backbone=dict(
      type="Plugin",
      module="/mnt/wangqiang/server/10.168.100.21/ai/nengwp/TCT/dino/infer_wq.py:get_backbone",
      weight="/mnt/wangqiang/server/10.168.100.21/ai/nengwp/TCT/dino/pretrain/dinov2_exp3/training_742052/teacher_checkpoint.pth" 
    ),
    neck=dict(
      type="ABMIL", 
      input_dim=1152, 
      hidden_dim=1152, 
      attention="gated",
      dropout=0.1,
    ), 
  ), 
  decoder=dict(
    head=dict(
      type="MLP",
      input_dim=1152,
      hidden_dim=1536,
      bottleneck_dim=256, 
      output_dim=len(class_names),
      n_layer=3,
      act="gelu",
      norm="layer_norm",
      # norm="batch_norm_1d",
      dropout=0.1,
    ),
  ),
  loss=dict(
    type="MILLoss",
    atten_loss=False, 
  ), 
  solver=dict(
    type="torch.optim.AdamW",
    lr=1e-3,
    weight_decay=1e-2, 
    scheduler=dict(
      type="torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
      T_0=1800,
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

  visualizer=dict(
    type="MILVisualizer", 
    every_n_steps_on_train=10,
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
    sync_on_compute=False,
    num_classes=len(class_names),
    ), 
  auc=dict(
    type="torchmetrics.AUROC",
    task="multiclass",
    sync_on_compute=False,
    num_classes=len(class_names),
  ), 
  kappa=dict(
    type="torchmetrics.CohenKappa",
    task="multiclass",
    sync_on_compute=False,
    num_classes=len(class_names),
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
del datetime, hashlib, torch, copy

_PROJECT_ = "mil"

_COMMENT_ = """
1. use pretrained ViT-L/16 as backbone;
2. use ABMIL as neck;
3. use balance sample strategy;
4. use 20X magnification;
5. use full size region and crop subtiles by shrink;
6. use AdamW as optimizer;
7. use MILVisualizer;
8. cache embedding in test phase;
"""