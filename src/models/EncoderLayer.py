import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from FeedForward import FeedForwardNN
from Residual_network import ResidualLayer

class EncoderLayer(nn.Module):

    def __init__(self,self_attention_block:MultiHeadAttention,feed_forward_block:FeedForwardNN,dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residualnetwork = nn.ModuleList(
            [ResidualLayer(dropout),
            ResidualLayer(dropout)])
        
    def forward(self,x):
        x = self.residualnetwork[0](x,lambda x:self.self_attention_block(x,x,x))
        x = self.residualnetwork[1](x,self.feed_forward_block)
        return x   
