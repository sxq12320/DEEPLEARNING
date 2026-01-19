import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import numpy as np
import sys
import os 

class ScaledDotProductAttention(nn.Module):
    '''单头注意力机制模块
    Args:
        @   dropout: dropout比率
    Returns:
        @   output: 注意力输出张量 [batch_size , n_heads , len_q , d_v]
        @   attn: 注意力权重张量 [batch_size , n_heads , len_q , len_k]
    计算步骤:   
        @  1. Q*K^T
        @  2. /sqrt(d_k)
        @  3. mask
        @  4. softmax
        @  5. dropout
    '''
    def __init__(self , dropout = 0.1):
        super(ScaledDotProductAttention , self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self , Q , K , V , mask = None):
        '''前向传播函数
        Args:
            @   Q: Query张量 [batch_size , n_heads , len_q , d_k]
            @   K: Key张量 [batch_size , n_heads , len_k , d_k]
            @   V: Value张量 [batch_size , n_heads , len_v , d_v]
            @   mask: 掩码张量 [batch_size , n_heads , len_q , len_k]
        Returns:
            @   output: 注意力输出 [batch_size , n_heads , len_q , d_v]
            @   attn: 注意力权重张量 [batch_size , n_heads , len_q , len_k]
        '''
        out_matmul = torch.matmul(Q , K.transpose(-2,-1))
        dk = K.size(-1)
        out_scaled = out_matmul / math.sqrt(dk)     

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)   # 扩展维度以匹配Q和K的形状
            elif mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0) # 扩展维度以匹配Q和K的形状
            out_scaled = out_scaled.masked_fill(mask == 0 , -1e9)       
        
        attn = self.softmax(out_scaled)
        attn_weight = self.dropout(attn)
        output = torch.matmul(attn_weight , V)

        return output , attn_weight
