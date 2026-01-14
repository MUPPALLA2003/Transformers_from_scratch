import torch
import torch.nn as nn
import LayerNormalization

class ResidualLayer(nn.Module):

    def __init__(self,dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self,x,sublayer):
        return self.norm(x+self.dropout(sublayer(x)))    
        