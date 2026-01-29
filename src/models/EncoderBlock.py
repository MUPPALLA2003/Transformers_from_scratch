import torch
import torch.nn as nn
import EncoderLayer
from LayerNormalization import LayerNormalization
from Embedding import Embeddings
from PositionalEncoding import PositionalEncoding
from MultiHeadAttention import MultiHeadAttention
from FeedForward import FeedForwardNN

class Encoder(nn.Module):

    def __init__(self,dropout:float,num_layers:int,d_model:int,d_ff:int,h:int,d_k:int):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            EncoderLayer.EncoderLayer(
                self_attention_block=MultiHeadAttention(d_model,h,dropout,d_k),
                feed_forward_block=FeedForwardNN(d_model,d_ff,dropout),
                dropout=dropout,
                d_model=d_model
                )
            for _ in range(self.num_layers)])
        self.norm = LayerNormalization(d_model)

    def forward(self,x,mask): 
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

             