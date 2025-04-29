############################## 1. DATASET
########## 1.1 DATA TRANSFORM

flip_and_color_jitter = [
  dict(type="RandomHorizontalFlip", p=0.5),
  dict(type="RandomVerticalFlip", p=0.5),
  # dict(type="RandomApply", transforms=[
  #   dict(type="ColorJitter", brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1), 
  # ], p=0.3), 
  # dict(type="RandomGrayscale", p=0.2),
]

quality = dict(
  type="RandomChoice", transforms=[
    dict(type="GaussianBlur", kernel_size=(5, 5), sigma=(0.5, 2.0)), 
    dict(type="RandomErasing", p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False),
  ], p=0.3
)

normalize = [
  dict(type="ToTensor"), 
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
    tile_size=1024, 
    tile_stride=896, 
    random_crop_patch_size=224, # random crop n patch from each tile
    random_crop_patch_num=16, 
    topk=0,
    shuffle=True, # shuffle patches
    padding_value=255,
    transforms=dict(
      type="TorchTransform", transforms=[*flip_and_color_jitter, *normalize],
    ),
  )]

val_view_transform = [
  dict(type="MILTransform", 
    tile_size=224, 
    tile_stride=224, 
    random_crop_patch_size=None, # random crop n patch from each tile
    random_crop_patch_num=None, 
    topk=0,
    shuffle=False,
    padding_value=255,
    transforms=dict(
      type="TorchTransform", transforms=normalize, 
    ), 
  )]

dataset_type = "SupervisedDataset"
class_names = ["NILM", "ASC-US", "LSIL", "ASC-H", "HSIL", "AGC"]
train_dataset = dict(
  type=dataset_type, 
  dataset=dict(
    type="SheetSource",
    sheet_file="/mnt/wangqiang/server/10.168.100.21/ai/internal/projects/hzztai/projects/tct/wsi_mil/data/v0.2数据集1654pkl.xlsx", 
    col_mapping={
      "path": "files",
      "label": "label",
      "用途": "phase", 
      "_SHEET_NAME_": "producer", 
    },
    # sheet_name=["BD", "Thinprep", "kfbio"],
    sheet_name=["kfbio"],
    filter=[
      ("phase", "==", "train"), 
    ], 
    apply=[
      ("files", "lambda file: file.replace('/root/workspace/server21/AI/ai_21/yujie', '/mnt/wangqiang/server/10.168.100.21/ai/yujie')"), 
      ("label", f"lambda label: {class_names}.index(label)"), 
    ]
  ),
  class_names=class_names, 
  std_func="lambda dataset: dict(files=dataset.pop('files'), annotations=dict(label=dataset.pop('label'), producer=dataset.pop('producer')))",
  transform=train_view_transform, 
  loader=loader, 
  desc="train tct wsi mil", 
  max_sample_num=None, 
  repeat=1,
  ann_keys=["label"],
  shuffle=True,
  cache=True,
)

val_dataset = dict(
  type=dataset_type, 
  dataset=dict(
    type="SheetSource",
    sheet_file="/mnt/wangqiang/server/10.168.100.21/ai/internal/projects/hzztai/projects/tct/wsi_mil/data/v0.2数据集1654pkl.xlsx", 
    col_mapping={
      "path": "files",
      "label": "label", 
      "用途": "phase", 
      "_SHEET_NAME_": "producer", 
    },
    # sheet_name=["BD", "Thinprep", "kfbio"],
    sheet_name=["kfbio"],
    filter=[
      ("phase", "==", "test"), 
    ], 
    apply=[
      ("files", "lambda file: file.replace('/root/workspace/server21/AI/ai_21/yujie', '/mnt/wangqiang/server/10.168.100.21/ai/yujie')"), 
      ("label", f"lambda label: {class_names}.index(label)"), 
    ]
  ),
  class_names=class_names, 
  std_func="lambda dataset: dict(files=dataset.pop('files'), annotations=dict(label=dataset.pop('label'), producer=dataset.pop('producer')))",
  transform=val_view_transform, 
  loader=loader, 
  desc="val tct wsi mil", 
  max_sample_num=None, 
  repeat=1,
  ann_keys=["label"],
  shuffle=False,
  cache=True,
)

########## 1.3 DATA LOADER
batch_size = 8
num_workers = 2
persistent_workers = True if num_workers > 0 else False

TRAIN_LOADER = dict(
  dataset=train_dataset,
  batch_size=batch_size,
  num_workers=num_workers,
  persistent_workers=False,
  # persistent_workers=persistent_workers,
  shuffle=True,
  pin_memory=True, 
  collate_fn=dict(
    type="AsymmetryShapeCollate",
  )
)

VALID_LOADER = dict(
  dataset=val_dataset,
  batch_size=batch_size,
  num_workers=num_workers,
  persistent_workers=False,
  # persistent_workers=persistent_workers,
  shuffle=False,
  pin_memory=True,
  collate_fn=dict(
    type="AsymmetryShapeCollate",
  )
)

############################## 2. MODEL
MODEL = dict(
  type="MIL", 
  embedding_group_size=10, 
  freeze_backbone=True,
  encoder=dict(
    backbone=dict(
      type="Plugin",
      module="/.slimai/plugins/standalone/get_embedding.py:get_backbone", 
    ),
    neck=dict(
      type="QMIL", 
      input_dim=768, 
      num_heads=12, 
      num_layers=3,
      act="gelu",
      norm="layer_norm",
      dropout=0.5,
    ), 
  ), 
  decoder=dict(
    head=dict(
      type="MLP",
      input_dim=768,
      hidden_dim=1024,
      bottleneck_dim=256, 
      output_dim=len(class_names),
      n_layer=2,
      act="gelu",
      norm="batch_norm_1d",
      dropout=0.5,
    ),
  ),
  loss=dict(
    type="MILLoss",
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
  max_epoch=300,
  compile=False, 
  checkpointing=True, 

  gradient=dict(
    amp=True, 
    accumulation_every_n_steps=1,
    clip=3.0, 
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
    eval_every_n_epochs=10, 
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
    ), 
  kappa=dict(
    type="torchmetrics.CohenKappa",
    task="multiclass",
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
del datetime, hashlib