import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
    
    def __init__(self,d_model:int,output_vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.output_vocab_size = output_vocab_size
        self.proj = nn.Linear(d_model,output_vocab_size)

    def forward(self,x):
        return torch.softmax(self.proj(x),dim=-1)    