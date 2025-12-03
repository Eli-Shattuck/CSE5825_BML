import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskViT(nn.Module):
    def __init__(
        self, in_channels=17, embed_dim=256, num_heads=8, depth=6, out_dim=128
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
            nn.Linear(embed_dim, out_dim),
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
