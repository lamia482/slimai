import numpy as np


class SampleStrategy(object):
  
  @classmethod
  def update_indices(cls, annotations, ann_keys, sample_strategy, indices) -> dict:
    assert (
      sample_strategy in ["random", "balance", "balance_up"]
    ), f"Sample strategy must be one of: {'random', 'balance', 'balance_up'}, but got: {sample_strategy}"
    
    if sample_strategy == "random":
      random_indices = indices.copy()
      np.random.shuffle(random_indices)
      return {ri: i for ri, i in zip(random_indices, indices)}

    elif sample_strategy == "balance":
      return cls.balance_indices(annotations, ann_keys, indices, keep_size=True)
    
    elif sample_strategy == "balance_up":
      return cls.balance_indices(annotations, ann_keys, indices, keep_size=False)
    
    else:
      indices = {i: j for i, j in enumerate(indices)}

    return indices

  @classmethod
  def balance_indices(cls, annotations, ann_keys, indices, freq_thres=0.5, keep_size=False):
    assert (
      "label" in ann_keys
    ), "Label key must be in ann_keys"

    labels = annotations["label"]

    # group indices by label
    labels_indices = {
      label: [index for index in indices if labels[index] == label]
      for label in set(labels)
    }

    # compute bincount, indices is not duplicate
    select_labels = [labels[i] for i in indices]
    bincount = np.bincount(select_labels)

    # compute max freq
    max_freq = bincount.max()
    freq_num = int(max_freq * freq_thres) # can be 0

    # balance indices by label
    balanced_indices = []
    for cid, indices in labels_indices.items():
      if len(indices) < freq_num:
        extra_num = freq_num - len(indices)
        extra_indices = np.random.choice(indices, size=extra_num, replace=True)
        indices = indices + extra_indices.tolist()
      balanced_indices.extend(indices)

    # balance_indices may be duplicate
    np.random.shuffle(balanced_indices)

    if keep_size:
      balanced_indices = balanced_indices[:len(labels)]
    
    return {i: j for i, j in enumerate(balanced_indices)}