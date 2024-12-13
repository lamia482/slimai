import os.path as osp
import numpy as np


########## 1. dataset transform ##########
backend_args=None

train_pipeline = [
  dict(type="LoadImageFromFile", backend_args=backend_args),
  dict(type="LoadAnnotations", with_bbox=True),
  dict(type="RandomChoiceResize", scales=[
    (1512, 1512), (1518, 1518), (1524, 1524), (1530, 1530), (1542, 1542), (1548, 1548), (1554, 1554), (1560, 1560)
  ] + [(1536, 1536)]*8, keep_ratio=True),
  dict(type="RandomCrop", crop_size=(640, 640), allow_negative_crop=True),
  dict(type="Resize", scale=None, scale_factor=1.0, keep_ratio=True),
  dict(type="RandomFlip", prob=0.5, direction="horizontal"),
  dict(type="RandomFlip", prob=0.5, direction="vertical"),
  dict(type="PackDetInputs", 
       meta_keys=["img_id", "img_path", "ori_shape", "img_shape",
                  "scale_factor", "flip", "flip_direction", "instances"])
]

test_pipeline = [
  dict(type="LoadImageFromFile", backend_args=backend_args),
  dict(type="LoadAnnotations", with_bbox=True),
  dict(type="Resize", scale=None, scale_factor=1.0, keep_ratio=True),
  dict(type="PackDetInputs",
       meta_keys=("img_id", "img_path", "ori_shape", "img_shape", 
                  "scale_factor", "instances"))
]

##########
classes = {
  "LSIL": "LSIL", 
  "AHSIL": ["ASC-H", "HSIL"],
  "AGC": ["AGC", "AGC-N"]
}
dataset_type = "SupervisedDataset"
dataset_dir = ""
ann_keys = ["labels", "masks", "instances", "texts"]

batch_size = 16
num_workers = 4
num_classes = len(classes)

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
    
val_dataloader = dict(
  _delete_=True, 
  batch_size=batch_size, 
  num_workers=num_workers, 
  pin_memory=False, 
  persistent_workers=True if num_workers > 0 else False, 
  drop_last=False, 
  sampler=dict(type="DefaultSampler", shuffle=False),
  dataset=dict(
    type=dataset_type,
    dataset=osp.join(dataset_dir, "valid.pkl"),
    ann_keys=ann_keys,
    transform=test_pipeline,
  )
)
    
test_dataloader = dict(
  _delete_=True, 
  batch_size=batch_size, 
  num_workers=num_workers, 
  pin_memory=False, 
  persistent_workers=True if num_workers > 0 else False, 
  drop_last=False, 
  sampler=dict(type="DefaultSampler", shuffle=False),
  dataset=dict(
    type=dataset_type,
    dataset=osp.join(dataset_dir, "test.pkl"),
    ann_keys=ann_keys,
    transform=test_pipeline,
  )
)

val_evaluator = dict(
  _delete_=True, 
  type="CocoMetric", 
  metric="bbox", 
  format_only=False, 
  classwise=True, 
)
    
test_evaluator = val_evaluator

##########

# 2. model settings
model = dict(
  type="ViT",
  num_classes=None,
  in_channels=3,
  embed_dims=768,
  num_layers=12,
  num_heads=12,
  mlp_ratio=4,
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

default_scope = "mmengine"

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
