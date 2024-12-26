import os.path as osp
import numpy as np


########## 1. dataset transform ##########
backend_args=None

view_pipeline = [
  dict(type="RandomResizedCrop", size=224, scale=(0.2, 1.)),
  dict(type="RandomGrayscale", prob=0.2, keep_channels=True),
  dict(
      type="ColorJitter",
      brightness=0.4,
      contrast=0.4,
      saturation=0.4,
      hue=0.4),
  dict(type="RandomFlip", prob=0.5),
]

train_pipeline = [
  dict(type="LoadImageFromFile", backend_args=backend_args),
  dict(type="MultiView", num_views=2, transforms=[view_pipeline]),
  dict(type="PackSelfSupInputs", meta_keys=["img_path"])
]

##########
dataset_type = "SupervisedDataset"
dataset_dir = ""
ann_keys = []

batch_size = 16
num_workers = 4

train_dataloader = dict(
  _delete_=True, 
  batch_size=batch_size, 
  num_workers=num_workers, 
  pin_memory=False, 
  persistent_workers=True if num_workers > 0 else False, 
  sampler=dict(type="DefaultSampler", shuffle=True),
  dataset=dict(
    type=dataset_type,
    dataset=osp.join(dataset_dir, "train.pkl"),
    ann_keys=ann_keys,
    transform=train_pipeline,
  )
)

##########

# 2. model settings
# model settings
model = dict(
  type="DINO",
  queue_len=65536,
  feat_dim=128,
  momentum=0.999,
  data_preprocessor=dict(
    mean=(123.675, 116.28, 103.53),
    std=(58.395, 57.12, 57.375),
    bgr_to_rgb=True),
  backbone=dict(
    type="ViT",
    arch="vit_base",
    patch_size=16,
    img_size=224, 
    use_lora=True, 
  ),
  neck=dict(
    type="DINONeck",
  ),
  head=dict(
    type="DINOHead",
  ),
)

##########

# training schedule
max_epochs = 100
max_epochs = max(1, max_epochs)
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# learning rate
base_lr = 0.001

param_scheduler = []
param_scheduler.append(
  dict(
    # and lr is updated by iteration
    # TODO: fix default scope in get function
    type="mmdet.QuadraticWarmupLR",
    by_epoch=True,
    begin=0,
    end=np.clip(max_epochs - 3, a_min=1, a_max=3),
    convert_to_iter_based=True
))

if 1 < max_epochs <= 6:
  param_scheduler.append(
    dict(
      type="CosineAnnealingLR",
      eta_min=base_lr * 0.05,
      begin=np.clip(max_epochs - 3, a_min=1, a_max=3),
      T_max=max_epochs,
      end=max_epochs,
      by_epoch=True,
      convert_to_iter_based=True
  ))

if max_epochs > 6:
  param_scheduler.append(
    dict(
      type="CosineAnnealingLR",
      eta_min=base_lr * 0.05,
      begin=np.clip(max_epochs - 3, a_min=1, a_max=3),
      T_max=max_epochs-1,
      end=max_epochs-1,
      by_epoch=True,
      convert_to_iter_based=True
  ))
  param_scheduler.append(
    dict(
      # use fixed lr during last 15 epochs
      type="ConstantLR",
      by_epoch=True,
      factor=1,
      begin=max_epochs-np.clip(max_epochs-6, a_min=1, a_max=3),
      end=max_epochs,
  ))


# optimizer
optim_wrapper = dict(
  _delete_=True, 
  type="OptimWrapper",
  optimizer=dict(type="SGD", lr=base_lr, momentum=0.9, weight_decay=5e-4, nesterov=True), 
  paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.),
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=64)

##########

default_scope = "mmdet"

default_hooks = dict(
  _delete_=True, 
  timer=dict(type="IterTimerHook"),
  logger=dict(type="LoggerHook", interval=10), 
  param_scheduler=dict(type="ParamSchedulerHook"),
  checkpoint=dict(type="CheckpointHook", interval=1),
  sampler_seed=dict(type="DistSamplerSeedHook"),
)
# yapf:enable
custom_hooks = [
  dict(type="NumClassCheckHook"), 
  dict(type="SyncNormHook", priority=48), 
  dict(
      type="EMAHook",
      ema_type="ExpMomentumEMA",
      momentum=0.0001,
      update_buffers=True,
      priority=49), 
]

env_cfg = dict(
  _delete_=True, 
  cudnn_benchmark=True,
  mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
  dist_cfg=dict(
    backend="nccl", 
    timeout=180000),
)

log_processor = dict(
  _delete_=True, 
  type="LogProcessor", window_size=50, by_epoch=True
)

log_level = "INFO"
load_from = None
resume = None

##########

# create model signature
from datetime import datetime
import hashlib
signature = datetime.now().strftime("%Y%m%d-{:s}".format(
  hashlib.md5(("+".join(map(str, (
    train_dataloader, model, param_scheduler, optim_wrapper, auto_scale_lr, load_from, resume
    )))).encode(encoding="UTF-8")
  ).hexdigest()[:8]
))
