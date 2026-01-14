import torch
import torch.nn as nn
import EncoderLayer
import LayerNormalization
import Embedding
import PositionalEncoding
import multiheadattention
import FeedForward

class EncoderBlock(nn.Module):

    def __init__(self,embedding:Embedding,positional_enc:PositionalEncoding,norm:LayerNormalization,dropout:float,num_layers:int,d_model:int,d_ff:int,h:int):
        super().__init__()
        self.embedding = embedding
        self.positional_enc = positional_enc
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            EncoderLayer(
                self_attention_block=multiheadattention(d_model,h,dropout),
                feed_forward_block=FeedForward(d_model,d_ff,dropout),
                dropout=dropout)
            for _ in range(num_layers)])
        self.norm = norm

    def forward(self,x):
        x = self.embedding(x)
        x = self.positional_enc(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

             