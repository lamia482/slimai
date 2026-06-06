import torch

from slimai.models.arch.mil import HierarchicalMIL


def test_compute_secondary_outputs_marginal_and_conditional():
  batch_size = 2
  num_primary = 2
  num_global = 3
  primary_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
  marginal_logits = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 2.0, 0.0],
  ])
  secondary_logits = {
    "h0": torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
    "h1": torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 0.0]]),
  }
  output = HierarchicalMIL._compute_secondary_outputs_static(
    marginal_logits=marginal_logits,
    primary_logits=primary_logits,
    secondary_logits=secondary_logits,
    primary_head_keys=["h0", "h1"],
    secondary_global_parent_idx=[0, 0, 1],
    secondary_global_local_idx=[0, 1, 1],
    global_secondary_num_classes=num_global,
    global_index_lookup={(0, 0): 0, (0, 1): 1, (1, 1): 2},
  )
  assert output["marginal_logits"].shape == (batch_size, num_global)
  assert output["conditional_logits"].shape == (batch_size, num_global)
  assert output["secondary_logits_flat"].shape == (batch_size, 5)
  assert torch.allclose(
    output["marginal_softmax"].sum(dim=-1),
    torch.ones(batch_size),
    atol=1e-5,
  )
  assert torch.allclose(
    output["conditional_softmax"].sum(dim=-1),
    torch.ones(batch_size),
    atol=1e-5,
  )
  assert output["marginal_labels"].tolist() == [0, 1]
  assert int(output["conditional_labels"][0]) == 0
  assert torch.allclose(
    output["conditional_softmax"].sum(dim=-1),
    torch.ones(batch_size),
    atol=1e-5,
  )
