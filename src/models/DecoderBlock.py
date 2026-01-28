import torch
import torch.nn as nn
from LayerNormalization import LayerNormalization
from Embedding import Embeddings
from PositionalEncoding import PositionalEncoding
from DecoderLayer import DecoderLayer
from MultiHeadAttention import MultiHeadAttention
from FeedForward import FeedForwardNN

class Decoder(nn.Module):

    def __init__(self,dropout: float,num_layers: int,d_model: int,d_ff: int,h: int,d_k:int):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                masked_self_attention_block=MultiHeadAttention(d_model, h, dropout,d_k),
                cross_attention_block=MultiHeadAttention(d_model, h, dropout,d_k),
                feed_forward_block=FeedForwardNN(d_model, d_ff, dropout),
                dropout=dropout,
                d_model=d_model
            )
            for _ in range(self.num_layers)
        ])

        self.norm = LayerNormalization(d_model)

    def forward(self,x,encoder_output,mask,tgt_mask):
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoder_output, mask, tgt_mask)
        return self.norm(x)
