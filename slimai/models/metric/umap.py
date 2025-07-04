import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
import umap
from slimai.helper.help_build import MODELS


@MODELS.register_module()
class UMAP(torch.nn.Module):
  def __init__(self, color_map):
    super().__init__()
    self.color_map = color_map
    return
  
  def forward(self, 
              embeddings: torch.Tensor, 
              targets: torch.Tensor) -> matplotlib.figure.Figure: # type: ignore
    obj = umap.UMAP(n_components=2, 
                           n_neighbors=200,
                           min_dist=0.5,
                           random_state=10482, 
                           metric='cosine')
    result = obj.fit_transform(embeddings)
    result = torch.tensor(result)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot()
    for label, (name, color) in enumerate(self.color_map.items()):
      mask = (targets == label)
      tmp = result[mask]
      ax.scatter(tmp[:, 0], tmp[:, 1], label=name, color=color)
    ax.legend(list(self.color_map.keys()))
    return fig
