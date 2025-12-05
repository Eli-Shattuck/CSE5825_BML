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


# class GATLayer(nn.Module):
#     def __init__(
#         self,
#         in_dim: int,
#         out_dim: int,
#         adj_matrix: torch.Tensor,
#     ):
#         super().__init__()
#         self.register_buffer("adj", adj_matrix)
#         self.W = nn.Linear(in_dim, out_dim, bias=False)
#         self.a = nn.Linear(2 * out_dim, 1, bias=False)
#         self.leakyrelu = nn.LeakyReLU(0.2)
#
#         nn.init.xavier_uniform_(self.W.weight.data)
#         nn.init.xavier_uniform_(self.a.weight.data)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, N, C = x.shape
#
#         # 1. Linear Transformation
#         Wh = self.W(x)  # (B, 64, out_dim)
#
#         # 2. Attention Mechanism (Optimized)
#         # We need to compute attention scores for all connected pairs.
#         # Broadcast trick to compute all-pairs concatenation
#         Wh_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)  # (B, 64, 64, out_dim)
#         Wh_j = Wh.unsqueeze(1).repeat(1, N, 1, 1)  # (B, 64, 64, out_dim)
#
#         a_input = torch.cat([Wh_i, Wh_j], dim=-1)  # (B, 64, 64, 2*out_dim)
#         e = self.leakyrelu(self.a(a_input)).squeeze(-1)  # (B, 64, 64)
#
#         # 3. Mask with Adjacency Matrix (Critical Step)
#         # Sets attention to -inf where no edge exists
#         zero_vec = -9e15 * torch.ones_like(e)
#         attention = torch.where(self.adj > 0, e, zero_vec)
#
#         attention = F.softmax(attention, dim=-1)  # (B, 64, 64)
#
#         # 4. Message Passing
#         h_prime = torch.matmul(attention, Wh)  # (B, 64, out_dim)
#
#         return F.elu(h_prime)


class GATLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        adj_matrix: torch.Tensor,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.concat = concat

        # Merge input projection into one layer for speed (Q, K, V)
        self.qkv = nn.Linear(in_dim, 3 * out_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Prepare mask for SDPA (Scaled Dot Product Attention)
        # We need to convert 1.0/0.0 mask to 0.0/-inf for adding
        self.register_buffer("additive_mask", None)
        self.register_buffer("adj", adj_matrix)

    def forward(self, h):
        if self.adj is None or not isinstance(self.adj, torch.Tensor):
            raise ValueError("Adjacency matrix creation failed.")

        B, N, _ = h.shape

        # 1. Prepare Mask (One-time setup if not done)
        if self.additive_mask is None:
            # Convert binary adj (1=connect, 0=block) to additive mask (0=keep, -inf=block)
            self.additive_mask = torch.zeros_like(self.adj)
            self.additive_mask = self.additive_mask.masked_fill(
                self.adj == 0, float("-inf")
            )
            # Reshape for broadcasting: (1, 1, N, N)
            self.additive_mask = self.additive_mask.unsqueeze(0).unsqueeze(0)

        # 2. Optimized Projection
        # qkv: (B, N, 3 * H * D)
        qkv = self.qkv(h)
        # Split: (B, N, H, D)
        q, k, v = qkv.view(B, N, 3, self.num_heads, self.head_dim).permute(
            2, 0, 3, 1, 4
        )

        # 3. FAST ATTENTION (Triggers Fused Kernels)
        # passing is_causal=False because we provide a custom mask
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=self.additive_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        # out: (B, H, N, D) -> (B, N, H, D)
        out = out.transpose(1, 2)

        if self.concat:
            out = out.reshape(B, N, -1)
        else:
            out = out.mean(dim=2)  # Averaging heads

        return out
