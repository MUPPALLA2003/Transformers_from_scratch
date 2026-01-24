import torch
import torch.nn as nn
from LayerNormalization import LayerNormalization
from Embedding import Embeddings
from PositionalEncoding import PositionalEncoding
from DecoderLayer import DecoderLayer
from MultiHeadAttention import MultiHeadAttention
from FeedForward import FeedForwardNN

class Decoder(nn.Module):

    def __init__(self,embedding: Embeddings,positional_enc: PositionalEncoding,dropout: float,num_layers: int,d_model: int,d_ff: int,h: int,d_k:int):
        super().__init__()
        self.embedding = embedding
        self.positional_enc = positional_enc
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(
                masked_self_attention_block=MultiHeadAttention(d_model, h, dropout,d_k),
                cross_attention_block=MultiHeadAttention(d_model, h, dropout,d_k),
                feed_forward_block=FeedForwardNN(d_model, d_ff, dropout),
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.norm = LayerNormalization(d_model)

    def forward(self, x, encoder_output, tgt_mask, src_pad_mask):
        x = self.embedding(x)
        x = self.positional_enc(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_pad_mask)
        return self.norm(x)
