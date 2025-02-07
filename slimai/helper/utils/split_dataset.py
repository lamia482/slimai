import numpy as np
import torch


def split_dataset(dataset, split_rule, desc=None, seed=10482):
  assert (
    isinstance(split_rule, dict) and len(split_rule) > 0
  ), "Split rule must be a non-empty dictionary, phase:rule[file,ratio,indice]"

  phases, rules = zip(*split_rule.items())
  assert (
    all(map(type, rules))
  ), "type of rule must be the same."

  final_indices_list = []

  if isinstance(rules[0], (int, float)):
    ratios = list(map(lambda r: r/np.sum(rules), rules))
    random_indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
    for ratio in ratios:
      split_indices = random_indices[:int(len(random_indices)*ratio)]
      random_indices = random_indices[int(len(random_indices)*ratio):]
      final_indices_list.append(split_indices)
  elif isinstance(rules[0], (tuple, list)) and isinstance(rules[0][0], int):
    for rule in rules:
      final_indices_list.append(rule)
  elif isinstance(rules[0], (tuple, list)) and isinstance(rules[0][0], str):
    for rule in rules:
      final_indices_list.append(dataset.files.index(rule))
  else:
    raise ValueError(f"Unsupported rule type: {type(rules[0])}")

  def wrap_dataset(indices, ds_desc):
    return type(dataset)(dataset.class_names, [dataset.files[i] for i in indices], 
                      annotations=dict(version=dataset.version, signature=dataset.signature, 
                                      labels=dataset.labels[indices]), 
                      transform=dataset.transform, to_rgb=dataset.to_rgb, desc=ds_desc)
  split_datasets = {
    phase: wrap_dataset(indices, dataset.desc+f"<Random split as {phase} with seed: {seed}>")
    for phase, indices in zip(phases, final_indices_list)
  }
  return split_datasets
