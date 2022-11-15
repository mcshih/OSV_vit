import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange
import math

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class cross_Attention(nn.Module):
    def __init__(self, hidden_size):
        super(cross_Attention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attention = Attention()
        self.ffn = PositionwiseFeedForward(hidden_size, hidden_size)

    def forward(self, query, key, value, mask=None, dropout=None):
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        output, att_map = self.attention(query, key, value)
        output = torch.flatten(output ,start_dim=1)

        return self.ffn(output), att_map # only output attention result