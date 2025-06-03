import cv2
import numpy as np
import torch
from itertools import chain
from . import scale_image


def put_gt_on_image(image, gt_instance, names, color=(255, 0, 0)):
  img = scale_image.to_batch_numpy_image(image)[0]

  if isinstance(gt_instance, (tuple, list)):
    boxes, labels = [gt["bbox"] for gt in gt_instance], [gt["bbox_label"] for gt in gt_instance]
  elif isinstance(gt_instance, dict):
    boxes, labels = gt_instance["boxes"], gt_instance["labels"]
  else:
    boxes, labels = gt_instance.bboxes, gt_instance.labels

  if isinstance(boxes, torch.Tensor) and isinstance(labels, torch.Tensor):
    boxes, labels = boxes.detach().cpu().numpy(), labels.detach().cpu().numpy()

  boxes, labels = np.array(boxes), np.array(labels)

  for index, ((x1, y1, x2, y2), cls_label) in enumerate(zip(
    boxes.astype("int"), labels.astype("int")
  )):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, 1)
    text = [
      f"{index}th:{names[cls_label]}"
    ]

    cv2.putText(img, "+".join(text), (x1, y1-5), 1, 1, color, 2)

  return img

def put_pred_on_image(image, pred_instance, names, score_thr=0.01, color=(0, 0, 255)):
  img = scale_image.to_batch_numpy_image(image)[0]

  if isinstance(pred_instance, (tuple, list)):
    boxes, scores, labels, *_ = pred_instance
  elif isinstance(pred_instance, dict):
    boxes, scores, labels = pred_instance["bboxes"], pred_instance["scores"], pred_instance["labels"]
  else:
    boxes, scores, labels = pred_instance.bboxes, pred_instance.scores, pred_instance.labels

  if isinstance(boxes, torch.Tensor):
    boxes, scores, labels = boxes.detach().cpu().numpy(), scores.detach().cpu().numpy(), labels.detach().cpu().numpy()

  indices = (scores >= score_thr)
  boxes, scores, labels = boxes[indices], scores[indices], labels[indices]

  for index, ((x1, y1, x2, y2), cls_label, cls_score) in enumerate(zip(
    boxes.astype("int"), labels.astype("int"), scores.tolist()
  )):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, 1)
    text = [
      f"{index}th:{names[cls_label]}:{cls_score:.3f}"
    ]

    cv2.putText(img, "+".join(text), (x1, y1-5), 2, 1, color)

  return img

def hstack_imgs(img_list, interval_width=10, interval_value=0):
  img_list = list(chain.from_iterable(
    zip(img_list, [interval_value+np.zeros([img_list[0].shape[0], interval_width, img_list[0].shape[2]]).astype("uint8")]*len(img_list))))[:-1]
  img = np.hstack(img_list)
  return img

def vstack_imgs(img_list, interval_width=10, interval_value=0):
  img_list = list(chain.from_iterable(
    zip(img_list, [interval_value+np.zeros([interval_width, img_list[0].shape[1], img_list[0].shape[2]]).astype("uint8")]*len(img_list))))[:-1]
  img = np.vstack(img_list)
  return img
