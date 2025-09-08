import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer Normalization class"""

    def __init__(self, embed_dim: int):
        """
        Class initialization
        
        Params:
            embed_dims (int): Embedding dimension
        
        """
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.tensor):
        """
        Forward class

        Params:
            x (torch.tensor): Input tensor

        Return:
            torch.tensor: Normalized tensor
        """
        mean = x.mean(dim=-1, keepdim=True)
        varience = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(varience+self.eps)
        return self.scale * norm_x + self.shift
    
if __name__ == "__main__":
    batch_example = torch.randn(2, 5)
    print(batch_example)
    ln = LayerNorm(embed_dim=5)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    print("Mean: ", mean)
    print("Variance: ", var)
    print(out_ln)
