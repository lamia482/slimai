############################## 1. DATASET
########## 1.1 DATA TRANSFORM
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
train_transform = A.Compose([
    A.OneOf([
      A.LongestMaxSize(256),
      A.Resize(256,256),
    ],p=0.8),
    A.PadIfNeeded(256,256,border_mode=cv2.BORDER_CONSTANT,value=(255,255,255)),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
      A.Blur(blur_limit=3, p=0.5),
      A.MotionBlur(blur_limit=3, p=0.5),
      A.AdvancedBlur(blur_limit=(3, 5), sigma_x_limit=(0.2, 0.5), sigma_y_limit=(0.2, 0.5), rotate_limit=30),
      A.Downscale(scale_min=0.75, scale_max=0.9, p=0.5),
      A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5)
    ], p=0.15),
    A.RandomCrop(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
  ])

test_transform = A.Compose([
    A.LongestMaxSize(256),
    A.PadIfNeeded(256,256,border_mode=cv2.BORDER_CONSTANT,value=(255,255,255)),
    A.CenterCrop(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
  ])

########## 1.2 DATA SET
CLASS_NAMES = (
  "ASC-US", "LSIL", "ASC-H", "HSIL", "AGC-N", 
  "TRI", "FUNGI", "WS", "CC", "ACTINO", 
  "GEC", "NILM", "Glycogen", "Repair", 
  "Debris"
)
def std_func(dataset, phase):
  import mmengine, os.path as osp
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
    "annotations": dict(labels=list(map(lambda file: CLASS_NAMES.index(osp.basename(osp.dirname(file))), dataset)))
  }
  return dataset

dataset_type = "SupervisedDataset"
train_dataset = dict(
  type=dataset_type, 
  dataset="/mnt/wangqiang/server/10.168.100.21/ai/internal/projects/hzztai/projects/research/aigc/jupyter/pathology_fms/dataset_split.pkl",
  std_func=std_func, 
  ann_keys=["labels"],
  transform=train_transform, 
  desc="molagu-train"
)
valid_dataset = dict(
  type=dataset_type, 
  dataset="/mnt/wangqiang/server/10.168.100.21/ai/internal/projects/hzztai/projects/research/aigc/jupyter/pathology_fms/dataset_split.pkl",
  std_func=std_func, 
  ann_keys=["labels"],
  transform=test_transform, 
  desc="molagu-valid"
)
test_dataset = dict(
  type=dataset_type, 
  dataset="/mnt/wangqiang/server/10.168.100.21/ai/internal/projects/hzztai/projects/research/aigc/jupyter/pathology_fms/dataset_split.pkl",
  std_func=std_func, 
  ann_keys=["labels"],
  transform=test_transform, 
  desc="molagu-test"
)

########## 1.3 DATA LOADER
batch_size = 16
num_workers = 8
persistent_workers = True if num_workers > 0 else False
TRAIN_LOADER = dict(
  dataset=train_dataset,
  batch_size=batch_size,
  num_workers=num_workers,
  persistent_workers=persistent_workers,
  shuffle=True,
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
      arch="vit_base",
      patch_size=16,
      img_size=224,
      use_lora=False,
      pretrained_weight=None,
    ),
    neck=None,
  ), 
  decoder=dict(
    head=dict(
      type="torch.nn.Linear",
      in_features=768,
      out_features=len(CLASS_NAMES),
    ),
  ),
  loss=dict(
    type="CrossEntropyLoss",
  ), 
  solver=dict(
    type="AdamW",
    lr=1e-3,
    weight_decay=1e-2,
  )
)

############################## 3. Metric
METRIC = dict(
  ACC=dict(
    type="Accuracy",
    topk=(1, 5),
  ), 
  KAPPA=dict(
    type="Kappa",
    num_classes=len(CLASS_NAMES),
  ),
)

############################## 4. Logger
LOG_LEVEL = "INFO"
LOG_DIR = "./log"
LOG_FILE = "log.txt"

############################## ALL SET
# create model signature
from datetime import datetime
import hashlib
signature = datetime.now().strftime("%Y%m%d-{:s}".format(
  hashlib.md5(("+".join(map(str, (
    TRAIN_LOADER, MODEL
    )))).encode(encoding="UTF-8")
  ).hexdigest()[:8]
))
