import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention


class MultiHeadAttentionSimple(nn.Module):
    """
    Multi-head attention module
    
    Stack multiple Attention module to perform the multi-head attention. Each attention head performs
    attention mechanism independently and the results are later concatenated.
    """

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float=0.5, bias: bool=False, num_heads: int=4):
        super().__init__()
        self.heads = nn.ModuleList(
            [Attention(d_in, d_out, context_length, dropout, bias) for _ in range(num_heads)]
        )

    def forward(self, x: torch.tensor):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module

    The multi-head attention class integrates everything in a single class where the query, key and value
    are split into multiple heads by reshaping appropriately. The results are finally concatenated to obtain
    attention weights.
    """

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float=0.5, qkv_bias: bool=False, num_heads: int=4):
        super().__init__()
        self.num_heads = num_heads
        self.d_out = d_out
        self.head_dim = d_out // num_heads

        # Initialize query, key and value weights. Usually d_in=d_out but for understanding keep it separate
        self.W_query = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.W_key = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.W_value = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        
        # Initialize dropout
        self.dropout = nn.Dropout(p=dropout)

        # Initialize register buffer for masking
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: torch.tensor, mask: bool=True):
        batch_size, num_tokens, d_in = x.shape
        query = self.W_query(x)
        key = self.W_key(x)
        value = self.W_value(x)

        # Reshape (batch_size, num_tokens, d_out) -> (batch_size, num_tokens, num_heads, head_dim)
        query = query.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        key = key.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        value = value.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        # Transpose to (batch_size, num_heads, num_tokens, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_weights = (query @ key.transpose(-2, -1)) / key.shape[-1]**0.5
        # Masking tokens
        if mask:
            attn_weights = attn_weights.masked_fill(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Transpose back to (batch_size, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ value).transpose(1, 2)
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)

        return context_vec


if __name__ == "__main__":
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89, 0.25],
         [0.55, 0.87, 0.66, 0.25],
         [0.57, 0.85, 0.64, 0.25],
         [0.22, 0.58, 0.33, 0.25],
         [0.77, 0.25, 0.10, 0.25],
         [0.05, 0.80, 0.55, 0.25]]
    )
    batch = torch.stack((inputs, inputs, inputs), dim=0)
    mha = MultiHeadAttention(inputs.shape[1], inputs.shape[1], batch.shape[1], 0.0, False, 2)
    out = mha(batch)
    print(out.shape)
    print(out)
