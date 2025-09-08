import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Simple Attention Block as defined in "Attention is all you need" paper."""

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, qkv_bias: bool=False):
        """
        Class initialization
        
        Params:
            d_in (int): Input embedding size
            d_out (int): Output embedding size
            context_length (int): Represent the supported input size of the LLM
            dropout (float): 0-1 probability for drop out
            qkv_bias (bool): Bias for Q,K,V
        """
        super().__init__()
        self.d_in = d_in
        self.context_length = context_length

        # Initialize query, key and value weights. Usually d_in=d_out but for understanding keep it separate
        self.W_query = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.W_key = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.W_value = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)

        # Register buffer automatically moves the element to appropriate device.
        # Not included in model.parameters() since it is non-trainable but included in state_dict
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

        # Initialize dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=True):
        """
        Forward method
        
        Params:
            x (tensor): Input tensor
            mask (bool): Mask for Causal Attention
        
        Return:
            attn_weights (tensor): Attention weights
        """
        batch_size, num_tokens, d_in = x.shape
        assert d_in == self.d_in
        query = self.W_query(x)
        key = self.W_key(x)
        value = self.W_value(x)

        attn_weights = (query @ key.transpose(-2, -1)) /  key.shape[-1]**0.5
        print("attn_weight 1: ", attn_weights)
        if mask:
            # attn_weights = attn_weights.masked_fill(self.tril[:num_tokens, :num_tokens] == 0, float('-inf'))
            attn_weights = attn_weights.masked_fill(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
            print("attn_weight 2: ", attn_weights)
        attn_weights = F.softmax(attn_weights, dim=-1)
        print("attn_weight 3: ", attn_weights)
        attn_weights = self.dropout(attn_weights)
        print("attn_weight 4: ", attn_weights)
        attn_weights = attn_weights @ value
        print("attn_weight 5: ", attn_weights)
        return attn_weights

    
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
    attn = Attention(inputs.shape[1], inputs.shape[1], batch.shape[1], 0.0)
    out = attn(batch)
    print(out.shape)