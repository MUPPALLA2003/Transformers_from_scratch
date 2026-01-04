import torch
import torch.nn as nn

class Embeddings(nn.Module):

    def __init__(self,d_model:int,vocabulary_size:int,padding_idx:int):
        super().__init__()
        self.d_model = d_model
        self.vocabulary_size = vocabulary_size
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocabulary_size,d_model,padding_idx)

    def forward(self,x):
        return self.embedding(x)*torch.sqrt(self.d_model)    