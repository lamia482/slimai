from typing import Dict, List
import torch
from slimai.helper.help_build import MODELS, build_metric


@MODELS.register_module()
class DINOMetric(torch.nn.Module):
  def __init__(self, 
               class_names: List[str],
               classifier: Dict, 
               umap: Dict, 
               acc: Dict, 
               kappa: Dict):
    super().__init__()
    self.class_names = class_names
    num_classes = len(class_names)
    self.classifier = build_metric(dict(**classifier, k=num_classes))
    self.umap = build_metric(umap)
    self.acc = build_metric(dict(**acc, num_classes=num_classes))
    self.kappa = build_metric(dict(**kappa, num_classes=num_classes))
    return
  
  def forward(self, 
              embeddings: torch.Tensor, 
              targets: torch.Tensor) -> Dict[str, torch.Tensor]:
    fig = self.umap(embeddings.cpu(), targets.cpu()) # cpu for umap
    self.classifier.fit(embeddings, targets)
    preds = self.classifier.predict(embeddings)
    acc = self.acc(preds, targets)
    kappa = self.kappa(preds, targets)
    return dict(
      umap=fig, 
      acc=acc, 
      kappa=kappa
    )
