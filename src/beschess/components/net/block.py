import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GATLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        adj_matrix: torch.Tensor,
    ):
        super().__init__()
        self.adj = adj_matrix  # shape (64, 64)
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)

        nn.init.xavier_uniform_(self.W.weight.data)
        nn.init.xavier_uniform_(self.a.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # 1. Linear Transformation
        Wh = self.W(x)  # (B, 64, out_dim)

        # 2. Attention Mechanism (Optimized)
        # We need to compute attention scores for all connected pairs.
        # Broadcast trick to compute all-pairs concatenation
        Wh_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)  # (B, 64, 64, out_dim)
        Wh_j = Wh.unsqueeze(1).repeat(1, N, 1, 1)  # (B, 64, 64, out_dim)

        a_input = torch.cat([Wh_i, Wh_j], dim=-1)  # (B, 64, 64, 2*out_dim)
        e = self.leakyrelu(self.a(a_input)).squeeze(-1)  # (B, 64, 64)

        # 3. Mask with Adjacency Matrix (Critical Step)
        # Sets attention to -inf where no edge exists
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(self.adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=-1)  # (B, 64, 64)

        # 4. Message Passing
        h_prime = torch.matmul(attention, Wh)  # (B, 64, out_dim)

        return F.elu(h_prime)
