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
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.concat = concat

        self.qkv = nn.Linear(in_dim, 3 * out_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer("additive_mask", None)
        self.register_buffer("adj", adj_matrix)

    def forward(self, h):
        if self.adj is None or not isinstance(self.adj, torch.Tensor):
            raise ValueError("Adjacency matrix creation failed.")

        B, N, _ = h.shape

        if self.additive_mask is None:
            self.additive_mask = torch.zeros_like(self.adj)
            self.additive_mask = self.additive_mask.masked_fill(
                self.adj == 0, float("-inf")
            )
            self.additive_mask = self.additive_mask.unsqueeze(0).unsqueeze(0)

        qkv = self.qkv(h)
        q, k, v = qkv.view(B, N, 3, self.num_heads, self.head_dim).permute(
            2, 0, 3, 1, 4
        )

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=self.additive_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        out = out.transpose(1, 2)

        if self.concat:
            out = out.reshape(B, N, -1)
        else:
            out = out.mean(dim=2)

        return out


class InterpretableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=False,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # 1. Initialize params to match standard PyTorch layer naming
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation function helper
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "silu" or activation == "swish":
            self.activation = F.silu
        else:
            raise ValueError(f"Activation {activation} not supported in this snippet")

        self.norm_first = norm_first

        # Placeholder to store the most recent attention weights
        # Shape: (Batch, Num_Heads, Seq_Len, Seq_Len)
        self.last_attn_weights = None

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        # We override this block to capture weights
        x, weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,  # VITAL: Ask for weights
            average_attn_weights=False,  # VITAL: Keep heads separate
            is_causal=is_causal,
        )
        self.last_attn_weights = weights  # <--- Capture happens here
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = src

        # Branch 1: Pre-Normalization (norm_first=True)
        # x = x + attn(norm1(x))
        # x = x + ffn(norm2(x))
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x = x + self._ff_block(self.norm2(x))

        # Branch 2: Post-Normalization (norm_first=False, PyTorch default)
        # x = norm1(x + attn(x))
        # x = norm2(x + ffn(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            )
            x = self.norm2(x + self._ff_block(x))

        return x
