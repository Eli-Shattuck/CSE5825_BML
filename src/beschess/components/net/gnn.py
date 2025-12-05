import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from .block import GATLayer


class MultiTaskGAT(nn.Module):
    def __init__(
        self,
        in_channels: int = 17,
        hidden_dim: int = 128,
        out_dim: int = 128,
        depth: int = 4,
    ):
        super().__init__()

        # Generate Static Adjacency Matrix (Queen + Knight moves)
        self.register_buffer("adj", self._create_chess_adjacency())
        if self.adj is None or not isinstance(self.adj, torch.Tensor):
            raise ValueError("Adjacency matrix creation failed.")

        # Embed raw features
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # GAT Layers
        self.encoder = nn.ModuleList(
            [
                copy.deepcopy(GATLayer(hidden_dim, hidden_dim, self.adj, num_heads=8))
                for _ in range(depth)
            ]
        )

        self.metric_head = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def _create_chess_adjacency(self):
        """
        Creates an adjacency matrix for a chessboard graph where
        each square is connected to squares reachable by a knight or queen move.
        """
        knight_moves = [
            (-2, -1),
            (-2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        ]
        queen_directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        adj = torch.zeros(64, 64)
        for sq in range(64):
            rank, file = divmod(sq, 8)
            adj[sq, sq] = 1  # Self-loop
            # Knight moves
            for dr, df in knight_moves:
                r, f = rank + dr, file + df
                if 0 <= r < 8 and 0 <= f < 8:
                    target_sq = r * 8 + f
                    adj[sq, target_sq] = 1.0
            # Queen moves
            for dr, df in queen_directions:
                r, f = rank, file
                while True:
                    r += dr
                    f += df
                    if 0 <= r < 8 and 0 <= f < 8:
                        target_sq = r * 8 + f
                        adj[sq, target_sq] = 1.0
                    else:
                        break

        return adj

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)

        x = self.input_proj(x)

        for gat_layer in self.encoder:
            x = F.elu(x + gat_layer(x))

        # Global Readout: Max Pooling works best for Tactics (finding the "sharpest" square)
        # Mean Pooling works best for Positional evaluation
        graph_embedding, _ = torch.max(x, dim=1)

        emb = F.normalize(self.metric_head(graph_embedding), p=2, dim=1)
        puzzle_logits = self.classifier_head(graph_embedding)

        return emb, puzzle_logits
