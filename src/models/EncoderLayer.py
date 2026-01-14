import torch
import torch.nn as nn
import multiheadattention
import FeedForward
import Residual_network

class EncoderLayer(nn.Module):

    def __init__(self,self_attention_block:multiheadattention,feed_forward_block:FeedForward,dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residualnetwork = nn.ModuleList(
            [Residual_network(dropout),
            Residual_network(dropout)])
        
    def forward(self,x):
        x = self.residualnetwork[0](x,lambda x:self.self_attention_block(x,x,x))
        x = self.residualnetwork[1](x,self.feed_forward_block)
        return x   
