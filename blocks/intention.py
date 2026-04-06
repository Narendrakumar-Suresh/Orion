import torch.nn as nn

class Intention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.l1 = nn.Linear(out_dim, in_dim, bias=False)
        self.l2 = nn.Linear(in_dim, out_dim, bias=False)
    
    def inject(self, z): return self.l1(z)[:, None, :]
    def encode(self, hidden): return self.l2(hidden[:, 0, :])