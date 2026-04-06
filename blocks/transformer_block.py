import torch.nn as nn
from .gqa import GQA
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, num_heads: int = 8, num_kv_heads: int = 4):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.attn = GQA(hidden_dim, num_heads, num_kv_heads)
        
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.ff1 = nn.Linear(hidden_dim, ffn_dim)
        self.ff1_gate = nn.Linear(hidden_dim, ffn_dim)
        self.ff2 = nn.Linear(ffn_dim, hidden_dim)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        
        normed = self.norm2(x)
        ff_out = self.ff2(F.silu(self.ff1_gate(normed)) * self.ff1(normed))
        x = x + ff_out
        
        return x