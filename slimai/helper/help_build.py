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


def compose_components(components, 
                       *, 
                       source, 
                       recursive_key=None):
  if not isinstance(source, (tuple, list)):
    source = [source]
  # convert source to callable
  source = list(map(lambda s: s if callable(s) else (
    lambda name: (getattr(s, name, None) or s.get(name))), 
    source))
    
  def _recursive_compose(component):
    if isinstance(component, (tuple, list)):
      return list(map(_recursive_compose, component))
    
    if component is None:
      return None
    
    assert (
      isinstance(component, dict) and "type" in component
    ), "Component must be a dict with 'type' key, but got: {}".format(component)

    component = component.copy()
    component_type = component.pop("type")
    
    for get_component_cls in source:
      try:
        component_cls = get_component_cls(component_type)
      except:
        component_cls = None
      if component_cls is not None:
        break
    assert (
      component_cls is not None
    ), "Component: {} not found.".format(component_type)

    inner_components = None
    if recursive_key is not None:
      inner_components = component.pop(recursive_key, None)

    # no transforms means it self is a basic transform
    if inner_components is None:
      return component_cls(**component)
    
    if not isinstance(inner_components, (list, tuple)):
      inner_components = [inner_components]
    
    return component_cls(_recursive_compose(inner_components))
  
  if isinstance(components, (tuple, list)):    
    components = list(map(_recursive_compose, components))
  else:
    components = _recursive_compose(components)
  return components

def build_transform(cfg) -> Callable:
  transforms = compose_components(cfg, source=TRANSFORMS)
  return ComposeTransform(transforms)

def build_dataset(cfg) -> torch.utils.data.Dataset:
  return compose_components(cfg, source=DATASETS)

def build_dataloader(cfg) -> torch.utils.data.DataLoader:
  dataset = build_dataset(cfg.pop("dataset"))
  loader = torch.utils.data.DataLoader(dataset, **cfg)
  return loader

def build_model(cfg) -> torch.nn.Module:
  return compose_components(cfg, source=MODELS)

def build_loss(cfg) -> torch.nn.Module:
  return

def build_solver(cfg, module: torch.nn.Module):
  return compose_components(cfg, source=[OPTIMIZERS, torch.optim])

def build_metric(cfg):
  return compose_components(cfg, source=METRICS)
