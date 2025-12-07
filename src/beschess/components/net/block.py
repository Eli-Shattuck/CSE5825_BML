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

        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "silu" or activation == "swish":
            self.activation = F.silu
        else:
            raise ValueError(f"Activation {activation} not supported in this snippet")

        self.norm_first = norm_first

        self.last_attn_weights = None

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        bsz, tgt_len, embed_dim = x.shape
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads
        scale = head_dim**-0.5

        q, k, v = F.linear(
            x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias
        ).chunk(3, dim=-1)

        q = q.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)

        q = q.reshape(bsz * num_heads, tgt_len, head_dim)
        k = k.reshape(bsz * num_heads, tgt_len, head_dim)
        v = v.reshape(bsz * num_heads, tgt_len, head_dim)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) * scale

        if attn_mask is not None:
            attn_weights += attn_mask

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            mask = mask.repeat(1, num_heads, 1, 1).view(bsz * num_heads, 1, tgt_len)
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)

        weights_to_store = attn_weights.view(bsz, num_heads, tgt_len, tgt_len)

        if weights_to_store.requires_grad:
            weights_to_store.retain_grad()

        self.last_attn_weights = weights_to_store

        attn_weights_for_math = weights_to_store.view(bsz * num_heads, tgt_len, tgt_len)

        attn_output = torch.bmm(attn_weights_for_math, v)

        attn_output = attn_output.view(bsz, num_heads, tgt_len, head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.contiguous().view(bsz, tgt_len, embed_dim)

        x = F.linear(
            attn_output, self.self_attn.out_proj.weight, self.self_attn.out_proj.bias
        )

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

        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x = x + self._ff_block(self.norm2(x))

        else:
            x = self.norm1(
                x
                + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            )
            x = self.norm2(x + self._ff_block(x))

        return x
