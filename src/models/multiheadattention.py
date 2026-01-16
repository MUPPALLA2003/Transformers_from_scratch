import torch 
import torch.nn as nn

class MultiHeadAttention(nn.Module):

    def __init__(self,d_model:int,h:int,dropout:float,d_k:int):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model//h 
        self.dropout = nn.Dropout(dropout)
        self.Q_matrix = nn.Linear(d_model,d_k,bias = False)
        self.K_matrix = nn.Linear(d_model,d_k,bias = False)
        self.V_matrix = nn.Linear(d_model,d_k,bias = False)
        self.O_matrix = nn.Linear(d_model,d_model,bias = False)

    @staticmethod
    def attention(query,key,value,dropout:float):
        attention_scores = dropout((query @ key.transpose(-2, -1)) / torch.sqrt(query.size(-1)).softmax(dim = -1))
        return attention_scores @ value
    
    def forward(self,q,k,v):
        B,C,_ = q.shape
        query = self.Q_matrix(q)
        key = self.K_matrix(k)
        value = self.V_matrix(v)
        query = query.view(B,C,self.h,self.d_k).transpose(1,2)
        key = key.view(B,C,self.h,self.d_k).transpose(1,2)
        value = value.view(B,C,self.h,self.d_k).transpose(1,2)
        x = MultiHeadAttention.attention(query,key,value,self.dropout)
        return self.O_matrix(x)


        

