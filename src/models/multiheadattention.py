import torch 
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):

    def __init__(self,d_model:int,h:int,dropout:float,d_k:int):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model//h 
        self.dropout = nn.Dropout(dropout)
        self.Q_matrix = nn.Linear(d_model,d_model,bias = False)
        self.K_matrix = nn.Linear(d_model,d_model,bias = False)
        self.V_matrix = nn.Linear(d_model,d_model,bias = False)
        self.O_matrix = nn.Linear(d_model,d_model,bias = False)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill((mask == 0).to(scores.device), float('-inf'))
        attention_probs = scores.softmax(dim=-1)
        if dropout is not None:
            attention_probs = dropout(attention_probs)
        return attention_probs @ value
    
    def forward(self,q,k,v,mask):
        B,C,_ = q.shape
        query = self.Q_matrix(q)
        key = self.K_matrix(k)
        value = self.V_matrix(v)
        query = query.view(B,C,self.h,self.d_k).transpose(1,2)
        key = key.view(B,C,self.h,self.d_k).transpose(1,2)
        value = value.view(B,C,self.h,self.d_k).transpose(1,2)
        x = MultiHeadAttention.attention(query,key,value,mask,self.dropout)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)
        return self.O_matrix(x)


        

