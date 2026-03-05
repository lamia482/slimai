import argparse
import copy
import importlib
import json
import os
import os.path as osp
import random
import sys
from pathlib import Path

import mmengine
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import cv2


# Script lives at slimai/scripts/train_sft.py; repo root is parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
  sys.path.insert(0, REPO_ROOT.as_posix())

from slimai.helper import help_build
from slimai.models.feature import apply_lora_to_vit


K_DIM_DICT = {
  "UNI": 1024,
  "CytoKD": 768,
  "CervicalKF": 1152,
}

MODEL_IMPORT_DICT = {
  "UNI": ("uni", "UNIHelper"),
  "CytoKD": ("cyto_helper", "CytoKDHelper"),
  "CervicalKF": ("kfd_helper", "KFDHelper"),
}


class SFTDataset(Dataset):
  def __init__(self, data_frame, transform, color):
    super().__init__()
    self.df = data_frame.reset_index(drop=True)
    self.transform = transform
    self.color = color
    return

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    item = self.df.iloc[idx]
    image = cv2.imread(item["filepath"])
    if self.color == "GRAY":
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      image = cv2.merge([image, image, image])
    else:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return dict(
      image=self.transform(image),
      label=int(item["subclass"]),
    )


def parse_args():
  parser = argparse.ArgumentParser(description="Train SFT classifier with optional LoRA.")
  parser.add_argument("--dataset-excel", required=True, help="Excel file containing filepath/subclass columns.")
  parser.add_argument("--model-name", default="CervicalKF", choices=list(K_DIM_DICT.keys()))
  parser.add_argument("--model-repo", required=True, help="External repository path for model helper.")
  parser.add_argument("--color", default="GRAY", choices=["GRAY", "RGB"])
  parser.add_argument("--label-mapping-file", default=None, help="Optional json file of label mapping.")
  parser.add_argument("--use-focal-loss", action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument("--lora-r", type=int, default=8)
  parser.add_argument("--lora-alpha", type=float, default=1.0)
  parser.add_argument("--lora-dropout", type=float, default=0.0)
  parser.add_argument("--max-epoch", default="1x", help="Int or '{n}x'. 1x means 12 epochs.")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--num-workers", type=int, default=4)
  parser.add_argument("--save-dir", required=True, help="Directory to save best checkpoint.")
  parser.add_argument("--exp-tag", default=None, help="Optional experiment tag.")
  parser.add_argument("--seed", type=int, default=10482)
  parser.add_argument("--train-ratio", type=float, default=0.7)
  parser.add_argument("--valid-ratio", type=float, default=0.09)
  return parser.parse_args()


def set_random_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  return


def build_model(model_name, model_repo):
  module_name, helper_name = MODEL_IMPORT_DICT[model_name]
  if model_repo not in sys.path:
    sys.path.append(model_repo)
  module = importlib.import_module(module_name)
  helper = getattr(module, helper_name)
  feature_extractor, _ = helper.build_model()
  return feature_extractor


def get_loss(num_classes, use_focal_loss=False):
  cfg = dict(type="BasicClassificationLoss")
  if use_focal_loss:
    cfg["cls_loss"] = dict(
      type="SoftmaxFocalLoss",
      num_classes=num_classes,
      alpha=None,
      gamma=2.0,
      reduction="mean",
    )
  return help_build.build_loss(cfg)


def get_solver(model, t0):
  cfg = dict(
    type="torch.optim.AdamW",
    lr=1e-3,
    weight_decay=1e-2,
    scheduler=dict(
      type="torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
      T_0=t0,
      T_mult=1,
      eta_min=1e-5,
    ),
  )
  return help_build.build_solver(mmengine.ConfigDict(cfg), model.parameters())


def validate(loader, model, loss_fn, device):
  model.eval()
  with torch.inference_mode():
    target_stack, logits_stack = [], []
    for batch in tqdm(loader, desc="Validating", leave=False):
      targets, images = batch["label"], batch["image"]
      images = images.to(device)
      targets = targets.to(device)
      logits = model(images)
      logits_stack.append(logits.softmax(-1))
      target_stack.append(targets)

  logits = torch.cat(logits_stack)
  targets = torch.cat(target_stack)
  loss = sum(loss_fn(logits, targets).values())

  logits_np = logits.cpu().numpy()
  targets_np = targets.cpu().numpy()
  preds = logits_np.argmax(axis=1)
  acc = accuracy_score(targets_np, preds)

  auc_per_class = []
  num_classes = logits_np.shape[1]
  targets_one_hot = np.eye(num_classes)[targets_np.astype(int)]
  for i in range(num_classes):
    try:
      auc_c = roc_auc_score(targets_one_hot[:, i], logits_np[:, i])
    except ValueError:
      auc_c = float("nan")
    auc_per_class.append(auc_c)
  auc = np.nanmean(auc_per_class)
  kappa = cohen_kappa_score(targets_np, preds)
  cm = confusion_matrix(targets_np, preds)
  return loss, acc, auc, auc_per_class, kappa, cm, logits_np, targets_np


def validate_and_log(loader, model, loss_fn, device, metrics, phase):
  loss, acc, auc, auc_per_class, kappa, cm, logits, targets = validate(loader, model, loss_fn, device)
  metrics[phase] = dict(
    loss=loss,
    acc=acc,
    auc=auc,
    auc_per_class=auc_per_class,
    kappa=kappa,
    cm=cm,
    logits=logits,
    targets=targets,
  )
  print(
    "{}: Loss: {:.4f}, Acc: {:.4f}, AUC: {:.4f}, Kappa: {:.4f}\nAUC per class: {}\nConfusion Matrix:\n{}".format(
      phase, loss, acc, auc, kappa, auc_per_class, cm
    )
  )
  return


def build_label_mapping(file_path):
  if file_path is None:
    return None
  with open(file_path, "r", encoding="utf-8") as fp:
    mapping = json.load(fp)
  return mapping


def map_label(value, mapping):
  if mapping is None:
    return value
  if value in mapping:
    return mapping[value]
  value_str = str(value)
  if value_str in mapping:
    return mapping[value_str]
  return None


def main():
  args = parse_args()
  set_random_seed(args.seed)

  if isinstance(args.max_epoch, str) and args.max_epoch.endswith("x"):
    max_epoch = int(args.max_epoch.replace("x", "")) * 12
  else:
    max_epoch = int(args.max_epoch)

  data_frame = pd.read_excel(args.dataset_excel)
  label_mapping = build_label_mapping(args.label_mapping_file)
  if label_mapping is not None:
    data_frame["subclass"] = data_frame["subclass"].apply(lambda v: map_label(v, label_mapping))
    data_frame = data_frame.dropna(subset=["subclass"])
    data_frame = data_frame[data_frame["subclass"] >= 0]
  data_frame["subclass"] = data_frame["subclass"].astype(int)

  holdout_ratio = max(0.0, min(1.0, 1 - args.train_ratio))
  train_df, temp_df = train_test_split(data_frame, test_size=holdout_ratio, random_state=args.seed)
  if len(temp_df) == 0 or args.valid_ratio <= 0:
    valid_df = temp_df
    test_df = temp_df
  else:
    valid_relative = min(0.999, args.valid_ratio / max(1e-6, holdout_ratio))
    valid_df, test_df = train_test_split(temp_df, test_size=1 - valid_relative, random_state=args.seed)

  normalize_op = [
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
  ]
  train_transform = transforms.Compose([transforms.RandomCrop([224, 224]), *normalize_op])
  test_transform = transforms.Compose([transforms.CenterCrop([224, 224]), *normalize_op])

  train_dataset = SFTDataset(train_df, train_transform, color=args.color)
  valid_dataset = SFTDataset(valid_df, test_transform, color=args.color)
  test_dataset = SFTDataset(test_df, test_transform, color=args.color)

  def wrap_dataloader(dataset, shuffle=False):
    return DataLoader(
      dataset,
      batch_size=args.batch_size,
      shuffle=shuffle,
      num_workers=args.num_workers,
      pin_memory=False,
    )

  train_loader = wrap_dataloader(train_dataset, shuffle=True)
  valid_loader = wrap_dataloader(valid_dataset, shuffle=False)
  test_loader = wrap_dataloader(test_dataset, shuffle=False)

  feature_extractor = build_model(args.model_name, args.model_repo)
  feature_extractor = apply_lora_to_vit(
    feature_extractor,
    r=args.lora_r,
    alpha=args.lora_alpha,
    dropout=args.lora_dropout,
  )

  num_classes = int(data_frame["subclass"].max()) + 1
  model = torch.nn.Sequential(
    feature_extractor,
    torch.nn.Linear(K_DIM_DICT[args.model_name], num_classes),
  )
  model.to(args.device)

  loss_fn = get_loss(num_classes=num_classes, use_focal_loss=args.use_focal_loss)
  solver, scheduler = get_solver(model, t0=max(1, len(train_loader)))

  best_model, best_epoch, min_loss = None, 0, float("inf")
  pbar = tqdm(total=len(train_loader))
  for epoch in range(max_epoch):
    pbar.reset()
    model.train()
    for step, batch in enumerate(train_loader):
      targets, images = batch["label"], batch["image"]
      images = images.to(args.device)
      targets = targets.to(args.device)
      logits = model(images)
      loss = sum(loss_fn(logits, targets).values())
      lr = scheduler.get_last_lr()[0]
      pbar.set_description(
        "Epoch: [{}/{}], Step: [{}/{}], LR: {:.6f}, Loss: {:.4f}".format(
          epoch + 1,
          max_epoch,
          step + 1,
          len(train_loader),
          lr,
          loss,
        )
      )
      pbar.update()
      solver.zero_grad()
      loss.backward()  # type: ignore
      solver.step()
      scheduler.step()

    valid_loss, acc, auc, auc_per_class, kappa, cm, *_ = validate(
      valid_loader,
      model,
      loss_fn,
      args.device,
    )
    if valid_loss <= min_loss:
      min_loss = valid_loss
      best_model = copy.deepcopy(model.state_dict())
      best_epoch = epoch
    print(
      "VALID - Epoch: [{}/{}]\nLoss: {:.4f}(min: {:.4f}), Acc: {:.4f}, AUC: {:.4f}, Kappa: {:.4f}\nAUC per class: {}\nConfusion Matrix:\n{}".format(
        epoch + 1,
        max_epoch,
        valid_loss,
        min_loss,
        acc,
        auc,
        kappa,
        auc_per_class,
        cm,
      )
    )
  pbar.close()

  if best_model is not None:
    model.load_state_dict(best_model)
    os.makedirs(args.save_dir, exist_ok=True)
    default_exp_tag = (
      f"{args.model_name}_{args.color}-FOCAL_LOSS_{'ON' if args.use_focal_loss else 'OFF'}-"
      f"SFT_LORA_{args.lora_r}_{args.lora_alpha}_{args.lora_dropout}"
    )
    exp_tag = args.exp_tag or default_exp_tag
    save_path = osp.join(args.save_dir, f"{exp_tag}-best.pth")
    torch.save(best_model, save_path)
    print(f"Best model loaded from epoch {best_epoch + 1}, loss {min_loss:.4f}")
    print(f"Best weight saved to: {save_path}")

  metrics = dict(model=model, weight=copy.deepcopy(model.state_dict()))
  validate_and_log(train_loader, model, loss_fn, args.device, metrics, "Train")
  validate_and_log(valid_loader, model, loss_fn, args.device, metrics, "Valid")
  validate_and_log(test_loader, model, loss_fn, args.device, metrics, "Test")
  return


if __name__ == "__main__":
  main()
