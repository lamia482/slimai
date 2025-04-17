import math
import torch
import torchvision.models as models
from slimai.helper.help_build import MODELS


__all__ = [
  "ViT", 
  "FlexViT", 
]

@MODELS.register_module()
class ViT(torch.nn.Module):
  def __init__(self, *, 
               arch="base", 
               patch_size=16, 
               image_size=224, 
               drop_head=False, 
               **kwargs):
    super().__init__()
    vit_param = FlexViT.vit_arch.get(arch, None)
    if vit_param is None:
      raise ValueError(f"ViT with arch {arch} is not supported from FlexViT")
    vit_param.update(dict(
      patch_size=patch_size, 
      image_size=image_size, 
      **kwargs
    ))

    self.embed_dim = vit_param["embed_dim"]
    self.vit = FlexViT.build_vit(**vit_param)

    if drop_head:
      self.vit.heads.head = torch.nn.Identity()
    return
  
  def forward(self, x):
    return self.vit(x)


class FlexViT(models.VisionTransformer):
  vit_arch = {
    "nano":  dict(image_size=224, patch_size=16, embed_dim=128, depth=3, num_heads=4, mlp_ratio=4.0), 
    "tiny":  dict(image_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0), 
    "small": dict(image_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0), 
    "base":  dict(image_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0), 
    "large": dict(image_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0), 
    "huge":  dict(image_size=224, patch_size=16, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4.0), 
    "giant": dict(image_size=224, patch_size=16, embed_dim=1536, depth=40, num_heads=24, mlp_ratio=4.0), 
  }

  @classmethod
  def build_vit(cls, *, 
                embed_dim, depth, num_heads, mlp_ratio, 
                image_size=224, patch_size=16, 
                dropout=0.1, attention_dropout=0.1, num_classes=1000, **kwargs):
    return cls(image_size=image_size, patch_size=patch_size, 
               num_layers=depth, num_heads=num_heads, 
               hidden_dim=embed_dim, mlp_dim=int(embed_dim*mlp_ratio), 
               dropout=dropout, attention_dropout=attention_dropout, 
               num_classes=num_classes, **kwargs)
    
  def _process_input(self, x: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape
    p = self.patch_size
    n_h = h // p
    n_w = w // p

    # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
    x = self.conv_proj(x)
    # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
    x = x.reshape(n, self.hidden_dim, n_h * n_w)

    # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
    # The self attention layer expects inputs in the format (N, S, E)
    # where S is the source sequence length, N is the batch size, E is the
    # embedding dimension
    x = x.permute(0, 2, 1)

    return x

  def forward(self, x: torch.Tensor):
    # Reshape and permute the input tensor
    B, nc, h, w = x.shape

    x = self._process_input(x)

    # Expand the class token to the full batch
    batch_class_token = self.class_token.expand(B, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)

    """.../site-packages/torchvision/models/vision_transformer.py:154
    # x = self.encoder(x)
    """
    torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")
    pos_embedding = self.interpolate_pos_encoding(self.encoder.pos_embedding, x, w, h)
    x = x + pos_embedding # [B, ~N, D] + [1, ~N, D] -> [B, ~N, D]
    x = self.encoder.ln(self.encoder.layers(self.encoder.dropout(x))) # [B, ~N, D] -> [B, ~N, D]

    # Classifier "token" as used by standard language architectures
    x = x[:, 0] # [B, ~N, D] -> [B, D]

    x = self.heads(x) # [B, D] -> [B, C]

    return x

  def interpolate_pos_encoding(self, pos_embed, x, w, h):
    npatch = x.shape[1] - 1
    N = pos_embed.shape[1] - 1
    if npatch == N and w == h:
      return pos_embed
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    dim = x.shape[-1]
    w0 = w // self.patch_size
    h0 = h // self.patch_size
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    w0, h0 = w0 + 0.1, h0 + 0.1
    patch_pos_embed = torch.nn.functional.interpolate(
        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
        mode='bicubic',
    )
    assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
