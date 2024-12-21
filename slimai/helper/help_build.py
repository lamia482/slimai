"""
Build components from configuration.

This function prioritizes customized components, followed by torch components,
and finally mmengine components.
"""
import torch
import mmengine
from typing import Callable, Union
from mmengine.registry import TRANSFORMS, DATASETS, MODELS, METRICS, OPTIMIZERS
from mmengine.dataset import Compose as ComposeTransform


def build_from_cfg(cfg: Union[mmengine.Config, dict], 
                   registry: mmengine.Registry):
  if cfg is None:
    return None
  
  # recursively handle list or tuple of element
  if isinstance(cfg, (list, tuple)):
    return list(map(lambda value: build_from_cfg(value, registry), cfg))
  
  assert (
    isinstance(cfg, dict)
  ), "cfg must be a dict"

  # recursively handle {name: element} dict-like cfg
  values = cfg.values()
  if all(isinstance(value, dict) for value in values):
    return {
      key: build_from_cfg(value, registry)
      for key, value in cfg.items()
    }
  
  # handle real element
  assert (
    "type" in cfg
  ), "cfg must have 'type' key, recursively check your config: \n{}".format(cfg)

  return registry.build(cfg)

def build_transform(cfg) -> Callable:
  transforms = build_from_cfg(cfg, TRANSFORMS)
  return ComposeTransform(transforms)

def build_dataset(cfg) -> torch.utils.data.Dataset:
  return build_from_cfg(cfg, DATASETS)

def build_dataloader(cfg) -> torch.utils.data.DataLoader:
  dataset = build_dataset(cfg.pop("dataset"))
  loader = torch.utils.data.DataLoader(dataset, **cfg)
  return loader

def build_model(cfg) -> torch.nn.Module:
  return build_from_cfg(cfg, MODELS)

def build_solver(cfg, module: torch.nn.Module):
  return build_from_cfg(cfg, OPTIMIZERS, module)

def build_metric(cfg):
  return build_from_cfg(cfg, METRICS)
