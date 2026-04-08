import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import RoPE

class GQA(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads

        self.wq = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_heads * self.head_dim, dim, bias=False)

        self.rope = RoPE(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape

        q = self.wq(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.wk(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        v = self.wv(x).reshape(B, T, self.num_kv_heads, self.head_dim)

        q, k = self.rope(q, k)

        k = torch.repeat_interleave(k, self.num_heads // self.num_kv_heads, dim=2)
        v = torch.repeat_interleave(v, self.num_heads // self.num_kv_heads, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.wo(out)