import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from ffn import FeedForward
from normalization import LayerNorm


class TransformerBlock(nn.Module):
    """Transformer Block used for building LLMs"""

    def __init__(self, cfg: dict):
        super().__init__()
        self.mha = MultiHeadAttention(
            d_in=cfg['embed_dim'],
            d_out=cfg['embed_dim'],
            context_length=cfg['context_length'],
            dropout=cfg['dropout'],
            qkv_bias=cfg['qkv_bias'],
            num_heads=cfg['num_heads']
        )
        self.ffn = FeedForward(cfg=cfg)
        self.norm1 = LayerNorm(embed_dim=cfg['embed_dim'])
        self.norm2 = LayerNorm(embed_dim=cfg['embed_dim'])
        self.dropout = nn.Dropout(cfg['dropout'])

    def forward(self, x: torch.tensor):
        shortcut = x
        x = self.norm1(x)
        x = self.mha(x)
        x = self.dropout(x)
        x += shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x += shortcut
        return x
    
if __name__ == "__main__":
    torch.manual_seed(123)
    
    cfg = {"vocab_size": 50257, # Vocabulary size
        "context_length": 1024, # Context length
        "embed_dim": 768,       # Embedding dimension
        "num_heads": 12,        # Number of attention heads
        "n_layers": 12,         # Number of layers
        "dropout": 0.1,         # Dropout rate
        "qkv_bias": False       # Query-Key-Value bias}
    }
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    tb = TransformerBlock(cfg=cfg)
    tb.to(device=device)
    x = torch.rand(2, 4, 768).to(device=device)
    print(tb(x).shape)


