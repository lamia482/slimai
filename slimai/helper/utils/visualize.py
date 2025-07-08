import cv2
import numpy as np
import torch
from itertools import chain
from . import scale_image, select


def put_gt_on_image(image, gt_instance, names, color=(255, 0, 0), no_text=False):
  img = scale_image.to_batch_numpy_image(image)[0]

  if isinstance(gt_instance, (tuple, list)):
    bboxes, labels = [gt["bbox"] for gt in gt_instance], [gt["bbox_label"] for gt in gt_instance]
  elif isinstance(gt_instance, dict):
    bboxes, labels = gt_instance["bboxes"], gt_instance["labels"]
  else:
    bboxes, labels = gt_instance.bboxes, gt_instance.labels

  if isinstance(bboxes, torch.Tensor) and isinstance(labels, torch.Tensor):
    bboxes, labels = bboxes.detach().cpu().numpy(), labels.detach().cpu().numpy()

  bboxes, labels = np.array(bboxes), np.array(labels)

  for index, ((x1, y1, x2, y2), cls_label) in enumerate(zip(
    bboxes.astype("int"), labels.astype("int")
  )):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, 1)

    if not no_text:
      text = [
        f"{index}th:{names[cls_label]}"
      ]
      cv2.putText(img, "+".join(text), (x1, y1-5), 1, 1, color, 2)

  return img

def put_pred_on_image(image, pred_instance, names, score_thr=0.01, color=(0, 0, 255), no_text=False):
  img = scale_image.to_batch_numpy_image(image)[0]

  if isinstance(pred_instance, (tuple, list)):
    bboxes, scores, labels, *_ = pred_instance
  elif isinstance(pred_instance, dict):
    bboxes, scores, labels = pred_instance["bboxes"], pred_instance["scores"], pred_instance["labels"]
  else:
    bboxes, scores, labels = pred_instance.bboxes, pred_instance.scores, pred_instance.labels

  if isinstance(bboxes, torch.Tensor):
    bboxes = np.array(bboxes.detach().cpu().tolist())
    scores = np.array(scores.detach().cpu().tolist())
    labels = np.array(labels.detach().cpu().tolist())

  indices = (scores >= score_thr)
  bboxes, scores, labels = bboxes[indices], scores[indices], labels[indices]

  for index, ((x1, y1, x2, y2), cls_label, cls_score) in enumerate(zip(
    bboxes.astype("int"), labels.astype("int"), scores.tolist()
  )):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, 1)

    if not no_text:
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

def square_imgs(img_list, interval_width=10, interval_value=0):
  img_num = len(img_list)
  square_s = np.floor(img_num**0.5).astype("int")
  img_list = img_list[:square_s*square_s]

  kwargs = dict(
    interval_width=interval_width, interval_value=interval_value
  )

  v_list = []
  for row_list in select.chunks(img_list, square_s):
    h = hstack_imgs(row_list, **kwargs)
    v_list.append(h)
  v = vstack_imgs(v_list)
  return v