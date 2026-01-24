import torch
import torch.nn as nn
from EncoderBlock import Encoder
from DecoderBlock import Decoder
from Embedding import Embeddings
from PositionalEncoding import PositionalEncoding
from Projection import ProjectionLayer

class Transformer(nn.Module):
    
    def __init__(self,src_embed:Embeddings,tgt_embed:Embeddings,pos:PositionalEncoding,encoder:Encoder,decoder:Decoder,projection:ProjectionLayer):
        super().__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.pos = pos
        self.encoder = encoder
        self.decoder = decoder
        self.projection = projection

    def encode(self,src,pad_mask):
        src = self.src_embed(src)
        src = self.pos(src)
        return self.encoder(src,pad_mask)

    def decode(self,tgt,encoder_output,mask,pad_mask):
        tgt = self.src_embed(tgt)
        tgt = self.pos(tgt)
        return self.decode(tgt,encoder_output,mask,pad_mask) 
    
    def project(self,tgt):
        return self.projection_layer(tgt)
