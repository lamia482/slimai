"""
Build components from configuration.

This function prioritizes customized components, followed by torch components,
and finally mmengine components.
"""
import mmcv
import torch
from typing import Callable, List
from torch.utils.data.distributed import DistributedSampler
from mmengine.dataset import Compose as ComposeTransform
from mmengine.registry import Registry, TRANSFORMS, DATASETS, MODELS, OPTIMIZERS
IMPORT = Registry("import")
LOADERS = Registry("loaders")
SOURCES = Registry("sources")
from slimai.helper.help_utils import print_log, dist_env


def compose_components(components, 
                       *, 
                       source, 
                       recursive_key=None):
  """Compose components from configuration."""
  if not isinstance(source, (tuple, list)):
    source = [source]
  # convert source to callable
  source = list(map(lambda s: s if callable(s) else (
    lambda name: (getattr(s, name, None) or s.get(name))), 
    [IMPORT] + source))
    
  def _recursive_compose(component):
    if isinstance(component, (tuple, list)):
      return list(map(_recursive_compose, component))
    
    if component is None:
      return None
    
    assert (
      isinstance(component, dict)
    ), "Component must be a dict but got: {}".format(component)

    component = component.copy()
    component_type = component.pop("type", None)
    
    # if component_type is None, it means it is a dict[str, component]
    if component_type is None:
      return {k: _recursive_compose(v) for k, v in component.items()}
    
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

def build_loader(cfg) -> Callable:
  """Build loader from configuration."""
  if cfg is None:
    return mmcv.imread
  return compose_components(cfg, source=LOADERS)

def build_source(cfg) -> Callable:
  """Build source from configuration."""
  return compose_components(cfg, source=SOURCES)

def build_transform(cfg) -> Callable:
  """Build transform from configuration."""
  transforms = compose_components(cfg, source=TRANSFORMS)
  if not isinstance(transforms, list):
    transforms = [transforms]
  return ComposeTransform(transforms)

def build_dataset(cfg) -> torch.utils.data.Dataset:
  """Build dataset from configuration."""
  dataset = compose_components(cfg, source=DATASETS)
  print_log(f"Dataset {dataset}", level="INFO")
  return dataset

def build_dataloader(cfg) -> torch.utils.data.DataLoader:
  dataset = build_dataset(cfg.pop("dataset", None))
  if dataset is None:
    return None
  if dist_env.is_dist_initialized():
    assert (
      "sampler" not in cfg
    ), "Sampler is not allowed in dataloader when DDP is enabled"
    print_log("use torch.utils.data.DistributedSampler for DDP")
    cfg["sampler"] = DistributedSampler(dataset, 
                                        shuffle=cfg.pop("shuffle", False),
                                        seed=dist_env.global_rank, 
                                        num_replicas=dist_env.global_world_size, 
                                        rank=dist_env.global_rank)
    
  if cfg.get("pin_memory", True):
    cfg["pin_memory"] = True
    cfg["pin_memory_device"] = f"cuda:{dist_env.local_rank}"

  if collate_fn := cfg.pop("collate_fn", None):
    cfg["collate_fn"] = build_loader(collate_fn)

  loader = torch.utils.data.DataLoader(dataset, **cfg)
  
  if getattr(loader.sampler, "set_epoch", None) is None:
    loader.sampler.set_epoch = lambda epoch: None

  return loader

def build_model(cfg) -> torch.nn.Module:
  module = compose_components(cfg, source=MODELS)
  if module is None:
    module = torch.nn.Identity()
  return module

def build_loss(cfg) -> torch.nn.Module:
  loss = compose_components(cfg, source=MODELS)
  assert (
    isinstance(loss, torch.nn.Module)
  ), "Loss must be a torch.nn.Module, but got: {}".format(type(loss))
  return loss

def build_solver(cfg, params: List[torch.nn.Parameter]):
  cfg = cfg.copy()

  if cfg.get("scheduler", None) is None:
    cfg["scheduler"] = dict(type="torch.optim.lr_scheduler.LambdaLR", lr_lambda=lambda epoch: 1)
  scheduler = cfg.pop("scheduler")
  
  cfg.params = params
  solver = compose_components(cfg, source=OPTIMIZERS)

  scheduler.optimizer = solver
  scheduler = compose_components(scheduler, source=OPTIMIZERS, recursive_key="schedulers")
  solver.scheduler = scheduler
  return solver

def build_metric(cfg):
  return compose_components(cfg, source=MODELS)
