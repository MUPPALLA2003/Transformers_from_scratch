import torch
import torch.nn as nn
from Embedding import Embeddings
from PositionalEncoding import PositionalEncoding
from EncoderBlock import Encoder
from DecoderBlock import Decoder
from Projection import ProjectionLayer
from Transformer import Transformer

def build_transformer(src_vocab_size:int,tgt_vocab_size:int,src_seq_len:int,tgt_seq_len:int,d_model:int,N:int,h:int,dropout:float,d_ff:int):

    src_embed = Embeddings(d_model,src_vocab_size)
    tgt_embed = Embeddings(d_model,tgt_vocab_size)
    src_pos = PositionalEncoding(d_model,src_seq_len,dropout)
    tgt_pos = PositionalEncoding(d_model,tgt_seq_len,dropout)
    encoder = Encoder(dropout,N,d_model,d_ff,h)
    decoder = Decoder(dropout,N,d_model,d_ff,h)
    projection = ProjectionLayer(d_model,tgt_vocab_size)
    transformer = Transformer(src_embed,tgt_embed,src_pos,tgt_pos,encoder,decoder,projection)

    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    return transformer        
