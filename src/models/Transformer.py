import torch
import torch.nn as nn
from EncoderBlock import Encoder
from DecoderBlock import Decoder
from Embedding import Embeddings
from PositionalEncoding import PositionalEncoding
from Projection import ProjectionLayer

class Transformer(nn.Module):
    
    def __init__(self,src_embed:Embeddings,tgt_embed:Embeddings,src_pos:PositionalEncoding,tgt_pos:PositionalEncoding,encoder:Encoder,decoder:Decoder,projection:ProjectionLayer):
        super().__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.encoder = encoder
        self.decoder = decoder
        self.projection = projection

    def encode(self,src,pad_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,pad_mask)

    def decode(self,tgt,encoder_output,mask,tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decode(tgt,encoder_output,mask,tgt_mask) 
    
    def project(self,tgt):
        return self.projection_layer(tgt)
