############################## 1. DATASET
########## 1.1 DATA TRANSFORM

flip_and_color_jitter = [
  dict(type="RandomHorizontalFlip", p=0.5),
  dict(type="RandomVerticalFlip", p=0.5),
  dict(type="RandomApply", transforms=[
    dict(type="ColorJitter", brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1), 
  ], p=0.8), 
  dict(type="RandomGrayscale", p=0.2),
]

normalize = [
  dict(type="ToTensor"), 
  dict(type="Normalize", mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
]

loader = dict(
  type="ReadWsiLoader", 
  random_scale=[5, 10, 20, 20, 20, 40], 
  random_crop_size=[256, 512, 1024], 
  anchor_file=None,
)

view_transform = [
  dict(type="DINOTransform", 
    global_transforms=dict(
      type="TorchTransform", transforms=[
        dict(type="RandomResizedCrop", size=224, scale=(0.4, 1.)),
        *flip_and_color_jitter,
        *normalize,
      ]),
    global_ncrops=2, 

    local_transforms=dict(
      type="TorchTransform", transforms=[
        dict(type="RandomResizedCrop", size=96, scale=(0.05, 0.4)),
        *flip_and_color_jitter,
        *normalize,
      ]), 
    local_ncrops=8
  )]

dataset_type = "UnSupervisedDataset"
train_dataset = dict(
  type=dataset_type, 
  dataset=dict(
    type="LocalSource",
    path="/mnt/wangqiang/server/172.16.10.17/slice/uploads/TCT/已入库",
    ext=[".kfb"], 
  ),
  std_func=None,
  transform=view_transform, 
  loader=loader, 
  desc="train custom dino", 
  max_sample_num=None,
  repeat=1,
)

val_dataset_type = "SupervisedDataset"
val_dataset = dict(
  type=val_dataset_type, 
  dataset="/hzztai/toolbox/_debug_/cls_dataset.pkl",
  std_func=None,
  ann_keys=["label"],
  transform=dict(
    type="TorchTransform", 
    transforms=[
      dict(type="Resize", size=[224, 224]),
      *normalize,
    ],
  ), 
  loader=None, 
  desc="val custom dino", 
  max_sample_num=None, 
  repeat=1,
  shuffle=False,
)


########## 1.3 DATA LOADER
batch_size = 64
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
  dataset=val_dataset,
  batch_size=batch_size,
  num_workers=num_workers,
  persistent_workers=persistent_workers,
  shuffle=True,
  pin_memory=True,
)

############################## 2. MODEL
MODEL = dict(
  type="DINO",
  encoder=dict(
    backbone=dict(
      type="ViT",
      arch="huge", 
      patch_size=16,
      drop_head=True,
      dropout=0.1, 
      attention_dropout=0.1, 
    ),
    neck=None, 
  ), 
  decoder=dict(
    head=dict(
      type="MLP",
      input_dim=1280,
      hidden_dim=2048,
      bottleneck_dim=256, 
      output_dim=65536,
      n_layer=2,
      act="gelu",
      norm=None,
      dropout=0.1,
    ),
  ),
  loss=dict(
    type="DINOLoss",
    output_dim=65536,
    warmup_teacher_temp=0.04,
    warmup_teacher_temp_epochs=30,
    teacher_temp=0.04,
    student_temp=0.1,
    center_momentum=0.9,
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
  momentum_teacher=0.9998, # recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256
)

############################## 3. RUNNER

RUNNER = dict(
  max_epoch=300,

  gradient=dict(
    amp=True, 
    accumulation_every_n_steps=1,
    clip=3.0, 
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
  type="DINOMetric", 
  class_names=["ASC-US", "LSIL", "ASC-H", "HSIL", "AGC-N", 
               "TRI", "FUNGI", "WS", "CC", "ACTINO", 
               "GEC", "NILM", "Glycogen", "Repair", "Debris"], 
  classifier=dict(
    type="CosineSimilarityClassifier"
  ),
  umap=dict(
    type="UMAP", 
    color_map = {
      "ASC-US": '#FF0000',    # Red
      "LSIL": '#FF1493',      # DeepPink 
      "ASC-H": '#FF00FF',     # Magenta
      "HSIL": '#DA70D6',      # Orchid
      "AGC-N": '#800080',     # Purple
      "TRI": '#0000FF',       # Blue
      "FUNGI": '#4169E1',     # RoyalBlue
      "WS": '#00BFFF',        # DeepSkyBlue
      "CC": '#87CEEB',        # SkyBlue
      "ACTINO": '#98FB98',    # PaleGreen
      "GEC": '#90EE90',       # LightGreen
      "NILM": '#32CD32',      # LimeGreen
      "Glycogen": '#228B22',  # ForestGreen
      "Repair": '#006400',    # DarkGreen
      "Debris": '#808080'     # Gray
    }
  ), 
  acc=dict(
    type="torchmetrics.Accuracy",
    task="multiclass"
  ), 
  kappa=dict(
    type="torchmetrics.CohenKappa", 
    task="multiclass"
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