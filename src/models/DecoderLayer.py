import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from FeedForward import FeedForwardNN
from Residual_network import ResidualLayer

class DecoderLayer(nn.Module):

    def __init__(self,masked_self_attention_block:MultiHeadAttention,cross_attention_block:MultiHeadAttention,feed_forward_block:FeedForwardNN,dropout:float,d_model:int):
        super().__init__()
        self.masked_self_attention_block = masked_self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_network = nn.ModuleList([
            ResidualLayer(dropout,d_model),
            ResidualLayer(dropout,d_model),
            ResidualLayer(dropout,d_model)]
            )
        
    def forward(self,x,encoder_output,mask,tgt_mask):
        x = self.residual_network[0](x,lambda x:self.masked_self_attention_block(x,x,x,mask))    
        x = self.residual_network[1](x,lambda x:self.cross_attention_block(x,encoder_output,encoder_output,tgt_mask))
        x = self.residual_network[2](x,self.feed_forward_block)
        return x