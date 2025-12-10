from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import clean_state_dict
from .block import InterpretableTransformerEncoderLayer


class MultiTaskViT(nn.Module):
    def __init__(
        self,
        in_channels: int = 20,
        embed_dim: int = 256,
        num_heads: int = 8,
        depth: int = 6,
        out_dim: int = 128,
    ):
        super().__init__()

        self.patch_proj = nn.Linear(in_channels, embed_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 65, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            enable_nested_tensor=False,
        )

        self.metric_head = nn.Sequential(
            nn.Linear(embed_dim, out_dim, bias=False),
            # nn.Linear(embed_dim, out_dim),
        )

        self.classifier_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        b, _, _, _ = x.shape

        x = x.flatten(2).transpose(1, 2)
        x = self.patch_proj(x)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        x = self.encoder(x)

        cls_output = x[:, 0]

        embedding = self.metric_head(cls_output)
        embedding = F.normalize(embedding, p=2, dim=1)

        puzzle_logits = self.classifier_head(cls_output)

        return embedding, puzzle_logits


class MultiTaskViT2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 20,
        embed_dim: int = 256,
        num_heads: int = 8,
        depth: int = 6,
        out_dim: int = 128,
    ):
        super().__init__()

        self.patch_proj = nn.Linear(in_channels, embed_dim)

        self.file_embed = nn.Parameter(torch.randn(1, 8, embed_dim) * 0.02)
        self.rank_embed = nn.Parameter(torch.randn(1, 8, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.metric_head = nn.Sequential(
            nn.Linear(embed_dim, out_dim, bias=False),
        )

        self.classifier_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor):
        B, C, _, _ = x.shape

        x = x.permute(0, 2, 3, 1).reshape(B, 64, C)

        x = self.patch_proj(x)

        pos = (
            self.file_embed.view(1, 8, 1, -1) + self.rank_embed.view(1, 1, 8, -1)
        ).view(1, 64, -1)

        x = x + pos

        x = self.encoder(x)
        x = self.norm(x)

        global_feat = x.mean(dim=1)

        embeddings = self.metric_head(global_feat)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        puzzle_logits = self.classifier_head(global_feat)

        return embeddings, puzzle_logits


def get_interpretable_vit(
    in_channels: int = 20,
    embed_dim: int = 256,
    num_heads: int = 8,
    depth: int = 6,
    out_dim: int = 128,
    path_to_weights: Path | str | None = None,
) -> nn.Module:
    encoder_layer = InterpretableTransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=embed_dim * 4,
        dropout=0.1,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )

    model = MultiTaskViT(
        in_channels=in_channels,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
        out_dim=out_dim,
    )

    model.encoder = nn.TransformerEncoder(
        encoder_layer=encoder_layer,  # type: ignore
        num_layers=depth,
        enable_nested_tensor=False,
    )

    if path_to_weights is not None:
        checkpoint = torch.load(path_to_weights, map_location="cpu")
        state_dict = (
            checkpoint["model_state_dict"]
            if "model_state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(clean_state_dict(state_dict), strict=True)

    return model


def extract_attention_weights(model, x):
    model.eval()
    with torch.no_grad():
        embeddings, puzzle_logit = model(x)
        puzzle_probs = torch.sigmoid(puzzle_logit)

        all_layer_weights = []

        for layer in model.encoder.layers:
            all_layer_weights.append(layer.last_attn_weights.cpu())

        all_layer_weights = torch.stack(all_layer_weights)
        all_layer_weights = all_layer_weights.permute(1, 0, 2, 3, 4)

        return embeddings, puzzle_probs, all_layer_weights


def compute_attention_rollout(attention_weights):
    batch_size, num_layers, num_heads, seq_len, _ = attention_weights.shape

    device = attention_weights.device

    avg_attention = attention_weights.mean(dim=2)

    residual = torch.eye(seq_len, seq_len, device=device).unsqueeze(0).unsqueeze(0)
    augmented_attention = avg_attention + residual

    augmented_attention = augmented_attention / augmented_attention.sum(
        dim=-1, keepdim=True
    )

    joint_attention = augmented_attention[:, 0]

    for n in range(1, num_layers):
        joint_attention = torch.bmm(augmented_attention[:, n], joint_attention)

    return joint_attention


def compute_head_importance(model, x):
    for param in model.parameters():
        param.requires_grad = True

    model.eval()
    model.zero_grad()

    embeddings, _ = model(x)

    loss = embeddings.norm()
    loss.backward()

    head_scores = []

    for layer_idx, layer in enumerate(model.encoder.layers):
        attn = layer.last_attn_weights

        if attn.grad is not None:
            score = (attn * attn.grad).abs().sum(dim=(0, 2, 3))

            for head_idx, val in enumerate(score):
                head_scores.append((head_idx, layer_idx, val.item()))
        else:
            print(f"Warning: No gradients found for Layer {layer_idx}.")

    return sorted(head_scores, key=lambda x: x[2], reverse=True)


def compute_faithfulness_map(model, board_tensor):
    """
    Computes a 64-element array where each value is the 'Impact Score'
    of that square (how much the embedding shifts when the square is removed).
    """
    model.eval()
    device = next(model.parameters()).device
    board_tensor = board_tensor.to(device)

    with torch.no_grad():
        base_emb, _ = model(board_tensor)

    impact_scores = np.zeros(64)

    for i in range(64):
        occluded_board = board_tensor.clone()

        occluded_flat = occluded_board.flatten(start_dim=2)
        occluded_flat[:, :, i] = 0  # Zero out all channels for this square

        occluded_board = occluded_flat.view_as(board_tensor)

        with torch.no_grad():
            new_emb, _ = model(occluded_board)

        dist = torch.norm(base_emb - new_emb).item()
        impact_scores[i] = dist

    return impact_scores
