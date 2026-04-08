import torch.nn as nn
import torch
from .intention import Intention
from .transformer_block import TransformerBlock

class Orion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed=nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.intention= Intention(cfg.hidden_dim, cfg.z_dim)
        self.blocks=nn.ModuleList([            
            TransformerBlock(cfg.hidden_dim, cfg.ffn_dim)
            for _ in range(cfg.num_layers)
        ])
        self.norm = nn.RMSNorm(cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size,bias=False)
    
    def forward(self, x,z_prev):
        x=self.embed(x)
        z_token=self.intention.inject(z_prev)
        x=torch.cat([z_token,x],dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x[:, 1:, :])
        z_next = self.intention.encode(x)
        return logits, z_next