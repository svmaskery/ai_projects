import torch
import torch.nn as nn
from transformer_block import TransformerBlock
from normalization import LayerNorm


class GPTLikeLLM(nn.Module):
    """GPT2 like LLM model"""

    def __init__(self, cfg: dict):
        """
        Class initialization

        Params:
            cfg (dict): Dict like config
        """
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["embed_dim"])
        self.drop_emb = nn.Dropout(cfg["dropout"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["embed_dim"])
        self.out_head = nn.Linear(cfg["embed_dim"], cfg["vocab_size"], bias=False)

    def forward(self, token_idx: torch.tensor):
        """
        Forward function

        Params:
            token_idx (torch.tensor): Torch tensor containing token indices

        Returns:
            logists (torch.tensor): Output logits from the model
        """
        batch_size, seq_len = token_idx.shape
        tok_embeds = self.tok_emb(token_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=token_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
if __name__ == "__main__":
    torch.manual_seed(123)
    cfg = {"vocab_size": 50257, # Vocabulary size
        "context_length": 1024, # Context length
        "embed_dim": 768, # Embedding dimension
        "num_heads": 12, # Number of attention heads
        "n_layers": 12, # Number of layers
        "dropout": 0.1, # Dropout rate
        "qkv_bias": False # Query-Key-Value bias}
    }
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = GPTLikeLLM(cfg).to(device=device)
    batch = torch.tensor(
        [[6109, 3626, 6100, 345],
         [6109, 1110, 6622, 257]]
    ).to(device=device)
    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    print(out)