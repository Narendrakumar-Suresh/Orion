import torch
import torch.nn as nn
import torch.nn.functional as F

class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len)
        freqs = torch.outer(t, inv_freq)
        
        # register_buffer automatically moves these to the GPU
        self.register_buffer("cos", torch.cos(freqs)[None, :, None, :])
        self.register_buffer("sin", torch.sin(freqs)[None, :, None, :])

    def forward(self, q, k):
        T = q.shape[1]
        cos, sin = self.cos[:, :T, :, :], self.sin[:, :T, :, :]

        def rotate(x):
            x1, x2 = x[..., 0::2], x[..., 1::2]
            y1 = x1 * cos - x2 * sin
            y2 = x1 * sin + x2 * cos
            return torch.stack([y1, y2], dim=-1).reshape_as(x)

        return rotate(q), rotate(k)