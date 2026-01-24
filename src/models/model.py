import torch
import torch.nn as nn
from Embedding import Embeddings
from PositionalEncoding import PositionalEncoding
from EncoderBlock import EncoderBlock
from DecoderBlock import DecoderBlock
from Projection import ProjectionLayer
from Transformer import Transformer
from LayerNormalization import LayerNormalization

def build_transformer(src_vocab_size:int,tgt_vocab_size:int,seq_len:int,d_model:int,N:int,h:int,dropout:float,d_ff:int,d_k:int):

    src_embed = Embeddings(d_model,src_vocab_size)
    tgt_embed = Embeddings(d_model,tgt_vocab_size)
    pos = PositionalEncoding(d_model,seq_len,dropout)
    norm = LayerNormalization()
    encoder = EncoderBlock(src_embed,pos,norm,dropout,N,d_model,d_ff,h,d_k)
    decoder = DecoderBlock(tgt_embed,pos,dropout,N,d_model,d_ff,h,d_k)
    projection = ProjectionLayer(d_model,tgt_vocab_size)
    transformer = Transformer(src_embed,tgt_embed,pos,encoder,decoder,projection)

    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)

    return transformer        
