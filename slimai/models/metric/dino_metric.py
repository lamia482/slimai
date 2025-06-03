from typing import Dict, List
import torch
from slimai.helper.help_build import MODELS, build_metric


@MODELS.register_module()
class DINOMetric(torch.nn.Module):
  def __init__(self, 
               class_names: List[str],
               umap: Dict, 
               classifier: Dict, 
               acc: Dict, 
               kappa: Dict):
    super().__init__()
    self.class_names = class_names
    num_classes = len(class_names)
    self.umap = build_metric(umap)

    if "num_classes" not in classifier:
      classifier["num_classes"] = num_classes
    self.classifier = build_metric(dict(**classifier))

    if "num_classes" not in acc:
      acc["num_classes"] = num_classes
    self.acc = build_metric(dict(**acc))

    if "num_classes" not in kappa:
      kappa["num_classes"] = num_classes
    self.kappa = build_metric(dict(**kappa))
    return
  
  def forward(self, 
              output: Dict[str, torch.Tensor], 
              targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    embeddings = output["embedding"]
    labels = targets["label"]

    fig = self.umap(embeddings.cpu(), labels.cpu()) # cpu for umap
    self.classifier.fit(embeddings, labels)
    preds = self.classifier.predict(embeddings)
    acc = self.acc(preds, labels)
    kappa = self.kappa(preds, labels)
    return dict(
      umap=fig, 
      acc=acc, 
      kappa=kappa
    )
