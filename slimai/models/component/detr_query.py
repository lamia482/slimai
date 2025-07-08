import torch
from slimai.helper.help_build import MODELS


@MODELS.register_module()
class DETRQuery(torch.nn.Module):
  def __init__(self, 
               *, 
               input_dim, 
               num_heads, 
               num_layers, 
               num_query, 
               dropout=0.0,
               ):
    super().__init__()
    self.input_dim = input_dim
    self.num_query = num_query

    transformer_decoder_layer = torch.nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads, 
                                                                 dropout=dropout, batch_first=True)
    self.decoder = torch.nn.TransformerDecoder(transformer_decoder_layer, num_layers=num_layers)
    self.query = torch.nn.Parameter(torch.randn(1, num_query, input_dim), requires_grad=True)
    return
  
  def forward(self, x):
    x = x.view(x.size(0), -1, self.input_dim)
    query = self.query.repeat(x.size(0), 1, 1)
    query = self.decoder(query, x)
    return query
