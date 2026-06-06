import torch
import mmengine
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from slimai.helper.help_utils import print_log
from slimai.helper.help_build import MODELS, build_model, build_loss
from slimai.helper.utils import PytorchNetworkUtils, get_cacher
from slimai.helper import DataSample
from .cls_arch import ClassificationArch


SLIDE_ENCODER_INPUT_NAME = "embedding_arr"
SLIDE_ENCODER_OUTPUT_NAMES_MIL = [
  "bag_embedding",
  "attention_weights",
  "logits",
  "softmax",
]
SLIDE_ENCODER_OUTPUT_NAMES_HIERARCHICAL = [
  "bag_embedding",
  "attention_weights",
  "primary_logits",
  "primary_softmax",
  "secondary_logits_flat",
  "secondary_local_softmax_flat",
  "marginal_logits",
  "marginal_softmax",
  "conditional_logits",
  "conditional_softmax",
  "primary_label",
  "marginal_label",
  "conditional_label",
]

__all__ = [
  "MIL",
  "HierarchicalMIL",
  "SLIDE_ENCODER_INPUT_NAME",
  "SLIDE_ENCODER_OUTPUT_NAMES_MIL",
  "SLIDE_ENCODER_OUTPUT_NAMES_HIERARCHICAL",
]

@MODELS.register_module()
class MIL(ClassificationArch):
  """Multiple Instance Learning (MIL) architecture.
  
  This class implements a MIL architecture for processing bags of instances.
  It consists of a backbone for feature extraction, a neck for aggregating
  instance features into bag-level representations, and a head for classification.
  
  Args:
    encoder (dict): Configuration for the encoder, including backbone and neck components.
    decoder (dict): Configuration for the decoder, including the head component.
    loss (dict, optional): Configuration for the loss function.
    solver (dict, optional): Configuration for the optimizer.
    embedding_group_size (int): Number of instances to process together in the backbone.
      This helps manage memory usage for large bags.
    freeze_backbone (bool): Whether to freeze the backbone parameters during training.
  """
  def __init__(self, *, 
               backbone=None, 
               neck=None, 
               head=None, 
               loss=None, 
               solver=None, 
               embedding_group_size=1, 
               freeze_backbone=False,
               ):
    self.freeze_backbone = freeze_backbone
    super().__init__(backbone=backbone, neck=neck, head=head, loss=loss, solver=solver)
    if freeze_backbone:
      print_log("Freezing backbone.")
      PytorchNetworkUtils.freeze(self.model.backbone)
    self.embedding_group_size = embedding_group_size
    self.cacher = get_cacher()
    return

  def init_layers(self, backbone, neck, head) -> torch.nn.Module:
    """Initialize the model layers.
    
    Args:
      backbone (dict): Configuration for the backbone components.
      neck (dict): Configuration for the neck components.
      head (dict): Configuration for the head components.
      
    Returns:
      torch.nn.ModuleDict: Dictionary containing the backbone, neck, and head modules.
    """
    print_log(
      f"Using default `init_layers` in {self.__class__.__name__}",
      level="WARNING", warn_once=True
    )
    backbone = build_model(backbone)
    neck = build_model(neck) 
    head = build_model(head)
    
    return torch.nn.ModuleDict(dict(backbone=backbone, neck=neck, head=head))
  
  def _forward_tensor(self, 
                batch_data: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                return_flow: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]:
    """Forward pass for tensor input.
    
    Processes a batch of bags through the model. Each bag contains multiple instances.
    
    Args:
      batch_data: Input data containing bags of instances, shape (B, ~N, C, H, W)
                 where B is batch size, ~N is variable number of instances per bag.
      return_flow: If True, returns intermediate outputs from each component.
      
    Returns:
      If return_flow is True, returns a dictionary with outputs from backbone, neck, and head.
      Otherwise, returns only the head output (classification logits).
    """
    embedding_list = self._collect_embedding_list(batch_data)
    backbone = embedding_list
    neck, atten_weights = self.model.neck(backbone) # type: ignore # (B, D)
    head = self.model.head(neck) # type: ignore # (B, C)

    if return_flow:
      return dict(
        backbone=backbone,
        atten_weights=atten_weights,
        neck=neck,
        head=head,
      )
    else:
      return atten_weights, head # type: ignore

  def _forward_loss(self, 
              embedding_dict: Dict[str, torch.Tensor], 
              batch_info: DataSample) -> Dict[str, torch.Tensor]:
    backbone = embedding_dict["backbone"]
    atten_weights = embedding_dict["atten_weights"]
    cls_logits = embedding_dict["head"]
    cls_targets = batch_info.label # type: ignore
    loss = self.loss(backbone, atten_weights, cls_logits, cls_targets)
    return loss

  def _normalize_single_attention(self, attention: Any) -> torch.Tensor:
    if torch.is_tensor(attention):
      tensor = attention.detach().reshape(-1).float()
      return tensor

    if isinstance(attention, (list, tuple)):
      values: List[float] = []
      for value in attention:
        if value is None:
          values.append(float("nan"))
          continue
        if torch.is_tensor(value):
          tensor_value = value.detach().reshape(-1).float()
          if tensor_value.numel() == 0:
            values.append(float("nan"))
          else:
            values.append(float(tensor_value[0].item()))
          continue
        try:
          values.append(float(value))
        except Exception:
          values.append(float("nan"))
      return torch.tensor(values, dtype=torch.float32)

    if attention is None:
      return torch.zeros((0,), dtype=torch.float32)

    try:
      return torch.tensor([float(attention)], dtype=torch.float32)
    except Exception:
      return torch.tensor([float("nan")], dtype=torch.float32)

  def _normalize_batch_attentions(self, atten_weights: Any) -> List[torch.Tensor]:
    if not isinstance(atten_weights, (list, tuple)):
      return []
    normalized: List[torch.Tensor] = []
    for sample_attention in atten_weights:
      normalized.append(self._normalize_single_attention(sample_attention))
    return normalized

  def _postprocess(self, 
                  batch_data: Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]], 
                  batch_info: DataSample) -> DataSample: 
    if isinstance(batch_data, dict):
      atten_weights = batch_data["atten_weights"]
      cls_logits = batch_data["head"]
    else:
      atten_weights, cls_logits = batch_data
    
    super()._postprocess(cls_logits, batch_info)
    normalized_attentions = self._normalize_batch_attentions(atten_weights)
    batch_info.output.update(dict(
      patch_attentions=normalized_attentions,
      attentions=normalized_attentions,
      atten_weights=normalized_attentions,
    ))

    # Only compute topk/tailk when attention weights are tensors (e.g. ABMIL); WMIL returns list of lists
    if atten_weights and all(torch.is_tensor(al) for al in atten_weights):
      topk = 8*8
      atten_topk = [al.topk(k=min(topk, al.shape[-1]), dim=-1, largest=True) for al in atten_weights]
      atten_tailk = [al.topk(k=min(topk, al.shape[-1]), dim=-1, largest=False) for al in atten_weights]
      topk_scores, topk_indices = list(map(list, zip(*atten_topk)))
      tailk_scores, tailk_indices = list(map(list, zip(*atten_tailk)))
      batch_info.output.update(dict(
        topk_atten_indices=topk_indices,
        topk_atten_scores=topk_scores,
        tailk_atten_indices=tailk_indices,
        tailk_atten_scores=tailk_scores,
      ))

    return batch_info

  def _forward_backbone_grouped(self, images: torch.Tensor) -> torch.Tensor:
    group_size = self.embedding_group_size
    if group_size <= 0:
      group_size = len(images)
    output = []
    if self.freeze_backbone:
      for start in range(0, len(images), group_size):
        with torch.inference_mode():
          embedding = self.model.backbone(images[start:start + group_size]) # type: ignore
        output.append(embedding)
    else:
      for start in range(0, len(images), group_size):
        embedding = self.model.backbone(images[start:start + group_size]) # type: ignore
        output.append(embedding)
    return torch.cat(output, dim=0)

  def _as_embedding_tensor(self, embedding: Any) -> torch.Tensor:
    if torch.is_tensor(embedding):
      return embedding
    return torch.as_tensor(embedding, dtype=torch.float32)

  def _collect_embedding_list(self, batch_data) -> List[torch.Tensor]:
    meta = getattr(self, "meta")
    embedding_list = []
    for vis_image, use_cache, embedding, embedding_key, visual_key in zip(
      batch_data,
      meta["use_cache"],
      meta.pop("embedding"),
      meta["embedding_key"],
      meta["visual_key"],
    ):
      if (not use_cache) or (embedding is None):
        embedding = self._forward_backbone_grouped(vis_image)
      else:
        embedding = self._as_embedding_tensor(embedding)
      embedding_list.append(embedding)
      if not use_cache:
        continue
      if not self.cacher.has(embedding_key):
        self.cacher.put(embedding_key, embedding)
      if not self.cacher.has(visual_key):
        self.cacher.put(visual_key, vis_image)
    return embedding_list

  @torch.inference_mode()
  def predict(self, batch_data, batch_info):
    """Use batched training forward for infer; export graph is single-bag only."""
    output = self._forward_tensor(batch_data, return_flow=False)
    return self.postprocess(output, batch_info)

  def _check_backbone_exportable(self) -> None:
    backbone = self.model.backbone # type: ignore
    if isinstance(backbone, torch.nn.Identity):
      raise ValueError(
        "MODEL.backbone is Identity; configure PatchEncoderBackbone before export."
      )
    if not hasattr(backbone, "export_model"):
      raise ValueError(
        f"Backbone {type(backbone).__name__} does not implement export_model()."
      )
    return

  def _check_neck_exportable(self) -> None:
    neck = self.model.neck # type: ignore
    if not hasattr(neck, "export_model"):
      raise ValueError(
        f"Neck {type(neck).__name__} does not support export; use ABMIL."
      )
    return

  def _build_slide_export_module(self) -> torch.nn.Module:
    return _MilExport(self).eval()

  def export_artifacts(self, *, cache_dir=None, cfg=None) -> Dict[str, Any]:
    del cache_dir, cfg
    self._check_backbone_exportable()
    self._check_neck_exportable()
    backbone = self.model.backbone # type: ignore
    preprocess = {}
    if hasattr(backbone, "export_preprocess"):
      preprocess = backbone.export_preprocess()
    return dict(
      patch_encoder=backbone.export_model(),
      slide_encoder=self._build_slide_export_module(),
      preprocess=preprocess,
    )

  def export_model(self) -> torch.nn.Module:
    return self.export_artifacts()["slide_encoder"]


class _MilExport(torch.nn.Module):
  """Slide-level export for single-head MIL: ``embedding_arr [N,K]`` -> logits."""

  def __init__(self, arch: "MIL"):
    super().__init__()
    self.neck = arch.model.neck.export_model() # type: ignore
    self.head = arch.model.head.export_model() # type: ignore
    return

  def forward(self, embedding_arr: torch.Tensor):
    bag_embedding, attention_weights = self.neck(embedding_arr)
    logits = self.head(bag_embedding.unsqueeze(0)).squeeze(0)
    softmax = torch.softmax(logits, dim=0)
    return bag_embedding, attention_weights, logits, softmax


class _HierarchicalMilExport(torch.nn.Module):
  """Slide-level export graph: ``embedding_arr [N,K]`` -> multi-head outputs."""

  def __init__(self, arch: "HierarchicalMIL"):
    super().__init__()
    self.neck = arch.model.neck.export_model() # type: ignore
    self.primary_head = arch.model.primary_head.export_model() # type: ignore
    self.marginal_head = arch.model.marginal_head.export_model() # type: ignore
    self.secondary_heads = torch.nn.ModuleDict(
      {name: head.export_model() for name, head in arch.model.secondary_heads.items()} # type: ignore
    )
    self.primary_head_keys = list(arch.primary_head_keys)
    self.secondary_global_parent_idx = list(arch.secondary_global_parent_idx)
    self.secondary_global_local_idx = list(arch.secondary_global_local_idx)
    self.global_secondary_num_classes = int(arch.global_secondary_num_classes)
    self._global_index_lookup = dict(arch._global_index_lookup)
    return

  def forward(self, embedding_arr: torch.Tensor):
    bag_embedding, attention_weights = self.neck(embedding_arr)
    neck_batch = bag_embedding.unsqueeze(0)
    primary_logits = self.primary_head(neck_batch).squeeze(0)
    marginal_logits = self.marginal_head(neck_batch).squeeze(0)
    secondary_logits_map = {}
    for head_name in self.primary_head_keys:
      head_logits = self.secondary_heads[head_name](neck_batch).squeeze(0)
      secondary_logits_map[head_name] = head_logits.unsqueeze(0)

    secondary_output = HierarchicalMIL._compute_secondary_outputs_static(
      marginal_logits=marginal_logits.unsqueeze(0),
      primary_logits=primary_logits.unsqueeze(0),
      secondary_logits=secondary_logits_map,
      primary_head_keys=self.primary_head_keys,
      secondary_global_parent_idx=self.secondary_global_parent_idx,
      secondary_global_local_idx=self.secondary_global_local_idx,
      global_secondary_num_classes=self.global_secondary_num_classes,
      global_index_lookup=self._global_index_lookup,
    )
    primary_softmax = torch.softmax(primary_logits, dim=0)
    primary_label = primary_logits.argmax(dim=0, keepdim=True).to(torch.int64)
    return (
      bag_embedding,
      attention_weights,
      primary_logits,
      primary_softmax,
      secondary_output["secondary_logits_flat"].squeeze(0),
      secondary_output["secondary_local_softmax_flat"].squeeze(0),
      secondary_output["marginal_logits"].squeeze(0),
      secondary_output["marginal_softmax"].squeeze(0),
      secondary_output["conditional_logits"].squeeze(0),
      secondary_output["conditional_softmax"].squeeze(0),
      primary_label,
      secondary_output["marginal_labels"].squeeze(0).to(torch.int64),
      secondary_output["conditional_labels"].squeeze(0).to(torch.int64),
    )


@MODELS.register_module()
class HierarchicalMIL(MIL):
  def __init__(
    self,
    *,
    backbone=None,
    neck=None,
    primary_head=None,
    marginal_head=None,
    secondary_heads: Optional[Dict[str, dict]] = None,
    head=None,  # alias of primary_head
    loss=None,
    solver=None,
    embedding_group_size=1,
    freeze_backbone=False,
    secondary_global_parent_idx: Optional[List[int]] = None,
    secondary_global_local_idx: Optional[List[int]] = None,
    primary_head_keys: Optional[List[str]] = None,
  ):
    self.secondary_heads_cfg = secondary_heads or {}
    self.marginal_head_cfg = marginal_head
    self.secondary_global_parent_idx = list(secondary_global_parent_idx or [])
    self.secondary_global_local_idx = list(secondary_global_local_idx or [])
    self.primary_head_keys = list(primary_head_keys or [])
    self.global_secondary_num_classes = len(self.secondary_global_parent_idx)
    if primary_head is None:
      primary_head = head
    super().__init__(
      backbone=backbone,
      neck=neck,
      head=primary_head,
      loss=loss,
      solver=solver,
      embedding_group_size=embedding_group_size,
      freeze_backbone=freeze_backbone,
    )
    if len(self.primary_head_keys) == 0:
      self.primary_head_keys = sorted(list(self.secondary_heads_cfg.keys()))
    self._global_index_lookup = {
      (int(parent), int(local)): int(global_idx)
      for global_idx, (parent, local) in enumerate(
        zip(self.secondary_global_parent_idx, self.secondary_global_local_idx)
      )
    }
    return

  def init_loss(self, loss) -> torch.nn.Module:
    if loss is None:
      loss = dict(type="HierarchicalMILLoss")
    else:
      loss = loss.copy()
    secondary_num_classes = {
      name: int(cfg["output_dim"])
      for name, cfg in self.secondary_heads_cfg.items()
    }
    loss["secondary_num_classes"] = secondary_num_classes
    loss["global_secondary_num_classes"] = int(self.global_secondary_num_classes)
    return build_loss(loss)

  def init_layers(self, backbone, neck, head) -> torch.nn.Module:
    if head is None:
      raise ValueError("`primary_head` (or `head`) must be provided for HierarchicalMIL.")
    if len(self.secondary_heads_cfg) == 0:
      raise ValueError("`secondary_heads` must be provided for HierarchicalMIL.")
    if self.marginal_head_cfg is None:
      raise ValueError("`marginal_head` must be provided for HierarchicalMIL.")
    backbone_module = build_model(backbone)
    neck_module = build_model(neck)
    primary_head = build_model(head)
    marginal_head = build_model(self.marginal_head_cfg)
    secondary_heads = torch.nn.ModuleDict(
      {name: build_model(cfg) for name, cfg in self.secondary_heads_cfg.items()}
    )
    return torch.nn.ModuleDict(
      dict(
        backbone=backbone_module,
        neck=neck_module,
        primary_head=primary_head,
        marginal_head=marginal_head,
        secondary_heads=secondary_heads,
      )
    )

  def _forward_tensor(
    self,
    batch_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
    return_flow: bool = False
  ) -> Union[Tuple[Any, ...], Dict[str, Any]]:
    backbone = self._collect_embedding_list(batch_data)
    neck, atten_weights = self.model.neck(backbone) # type: ignore
    primary_logits = self.model.primary_head(neck) # type: ignore
    marginal_logits = self.model.marginal_head(neck) # type: ignore
    secondary_logits = {
      name: head(neck)
      for name, head in self.model.secondary_heads.items() # type: ignore
    }

    if return_flow:
      return dict(
        backbone=backbone,
        atten_weights=atten_weights,
        neck=neck,
        primary_logits=primary_logits,
        marginal_logits=marginal_logits,
        secondary_logits=secondary_logits,
      )
    return atten_weights, primary_logits, marginal_logits, secondary_logits

  def _build_slide_export_module(self) -> torch.nn.Module:
    return _HierarchicalMilExport(self).eval()

  @staticmethod
  def _compute_secondary_outputs_static(
    *,
    marginal_logits,
    primary_logits,
    secondary_logits,
    primary_head_keys,
    secondary_global_parent_idx,
    secondary_global_local_idx,
    global_secondary_num_classes,
    global_index_lookup,
  ):
    batch_size = int(primary_logits.shape[0])
    device = primary_logits.device
    dtype = primary_logits.dtype
    if global_secondary_num_classes <= 0:
      zeros = torch.zeros((batch_size,), dtype=torch.int64, device=device)
      empty = torch.zeros((batch_size, 0), dtype=dtype, device=device)
      return dict(
        secondary_logits_flat=empty,
        secondary_local_softmax_flat=empty,
        marginal_logits=empty,
        marginal_softmax=empty,
        conditional_logits=empty,
        conditional_softmax=empty,
        marginal_scores=torch.zeros((batch_size,), dtype=dtype, device=device),
        marginal_labels=zeros,
        conditional_scores=torch.zeros((batch_size,), dtype=dtype, device=device),
        conditional_labels=zeros,
      )

    secondary_logits_flat = torch.cat(
      [secondary_logits[k] for k in primary_head_keys],
      dim=-1,
    )
    secondary_local_softmax_flat = torch.cat(
      [torch.softmax(secondary_logits[k], dim=-1) for k in primary_head_keys],
      dim=-1,
    )

    marginal_softmax = torch.softmax(marginal_logits, dim=-1)
    marginal_scores, marginal_labels = marginal_softmax.max(dim=-1)

    primary_log_prob = F.log_softmax(primary_logits, dim=-1)
    secondary_log_prob = {
      k: F.log_softmax(v, dim=-1) for k, v in secondary_logits.items()
    }
    conditional_logits = torch.zeros(
      (batch_size, global_secondary_num_classes),
      dtype=dtype,
      device=device,
    )
    for global_idx, (parent_idx, local_idx) in enumerate(
      zip(secondary_global_parent_idx, secondary_global_local_idx)
    ):
      parent_key = primary_head_keys[int(parent_idx)]
      conditional_logits[:, global_idx] = (
        primary_log_prob[:, int(parent_idx)]
        + secondary_log_prob[parent_key][:, int(local_idx)]
      )
    conditional_softmax = torch.softmax(conditional_logits, dim=-1)
    conditional_scores, conditional_labels = conditional_softmax.max(dim=-1)

    return dict(
      secondary_logits_flat=secondary_logits_flat,
      secondary_local_softmax_flat=secondary_local_softmax_flat,
      marginal_logits=marginal_logits,
      marginal_softmax=marginal_softmax,
      conditional_logits=conditional_logits,
      conditional_softmax=conditional_softmax,
      marginal_scores=marginal_scores,
      marginal_labels=marginal_labels,
      conditional_scores=conditional_scores,
      conditional_labels=conditional_labels,
    )

  def _compute_secondary_outputs(
    self,
    *,
    marginal_logits,
    primary_logits,
    secondary_logits,
  ):
    return self._compute_secondary_outputs_static(
      marginal_logits=marginal_logits,
      primary_logits=primary_logits,
      secondary_logits=secondary_logits,
      primary_head_keys=self.primary_head_keys,
      secondary_global_parent_idx=self.secondary_global_parent_idx,
      secondary_global_local_idx=self.secondary_global_local_idx,
      global_secondary_num_classes=self.global_secondary_num_classes,
      global_index_lookup=self._global_index_lookup,
    )

  def _forward_loss(self, embedding_dict: Dict[str, Any], batch_info: DataSample) -> Dict[str, torch.Tensor]:
    if not hasattr(self.loss, "primary_loss"):
      raise ValueError("HierarchicalMIL requires HierarchicalMILLoss.")
    if not hasattr(self.loss, "resolve_secondary_loss"):
      raise ValueError("HierarchicalMIL requires HierarchicalMILLoss with per-head secondary losses.")

    primary_logits = embedding_dict["primary_logits"]
    marginal_logits = embedding_dict["marginal_logits"]
    secondary_logits = embedding_dict["secondary_logits"]
    primary_targets = batch_info.label # type: ignore
    secondary_global_targets = getattr(batch_info, "label_secondary", None)
    if secondary_global_targets is None:
      secondary_global_targets = torch.full_like(primary_targets, fill_value=-1)
    secondary_local_targets = getattr(batch_info, "label_secondary_local", None)
    if secondary_local_targets is None:
      secondary_local_targets = torch.full_like(primary_targets, fill_value=-1)

    primary_loss_value = self.loss.primary_loss(primary_logits, primary_targets) # type: ignore

    secondary_output = self._compute_secondary_outputs(
      marginal_logits=marginal_logits,
      primary_logits=primary_logits,
      secondary_logits=secondary_logits,
    )
    conditional_logits = secondary_output["conditional_logits"]

    global_valid_mask = secondary_global_targets >= 0
    if global_valid_mask.any():
      marginal_loss_value = self.loss.marginal_loss( # type: ignore
        marginal_logits[global_valid_mask],
        secondary_global_targets[global_valid_mask],
      )
      conditional_loss_value = self.loss.conditional_loss( # type: ignore
        conditional_logits[global_valid_mask],
        secondary_global_targets[global_valid_mask],
      )
    else:
      zero = torch.zeros((), dtype=primary_logits.dtype, device=primary_logits.device)
      marginal_loss_value = zero
      conditional_loss_value = zero

    local_aux_loss_sum = torch.zeros((), dtype=primary_logits.dtype, device=primary_logits.device)
    local_aux_valid_num = torch.zeros((), dtype=primary_logits.dtype, device=primary_logits.device)
    local_aux_valid_mask = torch.zeros_like(primary_targets, dtype=torch.bool)
    for parent_idx, parent_key in enumerate(self.primary_head_keys):
      parent_mask = (primary_targets == int(parent_idx))
      local_targets = secondary_local_targets[parent_mask]
      if local_targets.numel() == 0:
        continue
      logits = secondary_logits[parent_key][parent_mask]
      if logits.shape[1] <= 1:
        continue
      valid_mask = (local_targets >= 0)
      if not valid_mask.any():
        continue
      group_logits = logits[valid_mask]
      group_targets = local_targets[valid_mask]
      group_loss = self.loss.resolve_secondary_loss(parent_key)(group_logits, group_targets) # type: ignore
      group_num = torch.tensor(float(group_targets.shape[0]), dtype=primary_logits.dtype, device=primary_logits.device)
      local_aux_loss_sum += group_loss * group_num
      local_aux_valid_num += group_num
      parent_indices = parent_mask.nonzero(as_tuple=False).reshape(-1)
      local_aux_valid_mask[parent_indices[valid_mask]] = True
    if local_aux_valid_num.item() > 0:
      local_aux_loss_value = local_aux_loss_sum / local_aux_valid_num
    else:
      local_aux_loss_value = torch.zeros((), dtype=primary_logits.dtype, device=primary_logits.device)

    loss = self.loss( # type: ignore
      primary_loss_value=primary_loss_value,
      marginal_loss_value=marginal_loss_value,
      conditional_loss_value=conditional_loss_value,
      local_aux_loss_value=local_aux_loss_value,
      local_aux_valid_mask=local_aux_valid_mask,
    )
    return loss

  def _postprocess(self, batch_data, batch_info: DataSample) -> DataSample:
    if isinstance(batch_data, dict):
      atten_weights = batch_data["atten_weights"]
      primary_logits = batch_data["primary_logits"]
      marginal_logits = batch_data["marginal_logits"]
      secondary_logits = batch_data["secondary_logits"]
    else:
      atten_weights, primary_logits, marginal_logits, secondary_logits = batch_data

    primary_softmax = torch.softmax(primary_logits, dim=1)
    primary_scores = primary_softmax.max(dim=1).values
    primary_labels = primary_softmax.argmax(dim=1)
    secondary_output = self._compute_secondary_outputs(
      marginal_logits=marginal_logits,
      primary_logits=primary_logits,
      secondary_logits=secondary_logits,
    )

    batch_info.output = dict(
      logits=primary_logits,
      softmax=primary_softmax,
      scores=primary_scores,
      labels=primary_labels,
      label=dict(
        logits=primary_logits,
        softmax=primary_softmax,
        scores=primary_scores,
        labels=primary_labels,
      ),
      label_secondary=dict(
        logits=secondary_output["marginal_logits"],
        softmax=secondary_output["marginal_softmax"],
        scores=secondary_output["marginal_scores"],
        labels=secondary_output["marginal_labels"],
      ),
      label_secondary_conditional=dict(
        logits=secondary_output["conditional_logits"],
        softmax=secondary_output["conditional_softmax"],
        scores=secondary_output["conditional_scores"],
        labels=secondary_output["conditional_labels"],
      ),
      label_secondary_local=dict(
        logits=secondary_output["secondary_logits_flat"],
        softmax=secondary_output["secondary_local_softmax_flat"],
      ),
    )
    normalized_attentions = self._normalize_batch_attentions(atten_weights)
    batch_info.output.update(dict(
      patch_attentions=normalized_attentions,
      attentions=normalized_attentions,
      atten_weights=normalized_attentions,
    ))

    if atten_weights and all(torch.is_tensor(al) for al in atten_weights):
      topk = 8 * 8
      atten_topk = [al.topk(k=min(topk, al.shape[-1]), dim=-1, largest=True) for al in atten_weights]
      atten_tailk = [al.topk(k=min(topk, al.shape[-1]), dim=-1, largest=False) for al in atten_weights]
      topk_scores, topk_indices = list(map(list, zip(*atten_topk)))
      tailk_scores, tailk_indices = list(map(list, zip(*atten_tailk)))
      batch_info.output.update(dict(
        topk_atten_indices=topk_indices,
        topk_atten_scores=topk_scores,
        tailk_atten_indices=tailk_indices,
        tailk_atten_scores=tailk_scores,
      ))
    return batch_info
  