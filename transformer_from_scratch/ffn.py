import torch
import torch.nn as nn

class GELU(nn.Module):
    """Activation layer for the transformer block"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.tensor):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
            ))
    
class FeedForward(nn.Module):
    """Feed forward layers with GELU activation"""

    def __init__(self, cfg: dict):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["embed_dim"], 4 * cfg["embed_dim"]),
            GELU(),
            nn.Linear(4 * cfg["embed_dim"], cfg["embed_dim"]),
            )
        
    def forward(self, x: torch.tensor):
        return self.layers(x)