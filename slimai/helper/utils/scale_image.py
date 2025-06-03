import cv2
import mmcv
import torch
import numpy as np


def to_image(image):
  try:
    image = mmcv.imread(image)
  except:
    pass
  return image
    
def to_batch_tensor_image(image, device=None):
  """convert color numpy image([NHWC or HWC]) to tensor of NCHW"""
  image = to_image(image)
  
  if isinstance(image, np.ndarray):
    image = torch.from_numpy(image)

  if device is not None:
    image = image.to(device)

  assert (
    image.ndim in [3,4]
  ), f"Expect image in ndim of [3,4], but got: f{image.ndim}"

  if image.ndim == 3:
    image = image[None, ...]

  if (image.shape[-1] in [3,4]) and (image.shape[-1] < image.shape[1]) and (image.shape[-1] < image.shape[2]):
    image = image.permute([0,3,1,2]) # NHWC to NCHW
  return image

def to_batch_numpy_image(image):
  """convert data to numpy image, in NHWC"""
  image = to_image(image)
  
  if isinstance(image, torch.Tensor):
    image = image.cpu().detach().numpy()

  assert (
    image.ndim in [3,4]
  ), f"Expect image in ndim of [3,4], but got: f{image.ndim}"

  if image.ndim == 3:
    image = image[None, ...]

  if (image.shape[1] in [3,4]) and (image.shape[1] < image.shape[2]) and (image.shape[1] < image.shape[3]):
    image = image.transpose([0,2,3,1]) # NCHW to NHWC

  if 0 <= image.min() <= image.max() <= 1:
    image = image * 255

  image = cv2.convertScaleAbs(image)
  return image
