import torch

from slimai.models.component import vit as vit_module
from slimai.models.component.vit import ViT


class _FakeInnerViT(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.fc = torch.nn.Linear(3, 8)
    return

  def forward(self, x, cls_pooling=True):
    del cls_pooling
    return self.fc(x.mean(dim=(2, 3)))


def test_vit_export_matches_forward(monkeypatch):
  monkeypatch.setattr(vit_module.FlexViT, "build_vit", lambda **_kwargs: _FakeInnerViT())
  vit = ViT(
    arch="base",
    patch_size=16,
    image_size=224,
    drop_head=True,
    cls_pooling=True,
  )
  vit.eval()
  x = torch.randn(2, 3, 4, 4)
  export_module = vit.export_model()
  assert torch.allclose(export_module(x), vit(x), atol=1e-5)
