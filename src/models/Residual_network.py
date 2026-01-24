import torch
import torch.nn as nn
from LayerNormalization import LayerNormalization

class ResidualLayer(nn.Module):

    def __init__(self,dropout:float,d_model:int):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)

    def forward(self,x,sublayer):
        return self.norm(x+self.dropout(sublayer(x)))    
        