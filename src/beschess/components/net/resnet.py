import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

from .block import SEBlock


class SEResBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        out += residual
        out = self.act(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += residual
        out = self.act(out)
        return out


class BaseEmbeddingNet(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_blocks: int,
        block: nn.Module,
    ):
        super().__init__()
        self.conv_input = nn.Conv2d(17, 64, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(64)
        self.res_layers = nn.Sequential(
            *[copy.deepcopy(block) for _ in range(num_blocks)]
        )

        self.conv_output = nn.Conv2d(64, 32, 1)
        self.fc = nn.Linear(32 * 8 * 8, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn_input(self.conv_input(x)))
        out = self.res_layers(out)
        out = F.relu(self.conv_output(out))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = F.normalize(out, p=2, dim=1)
        return out


class ResEmbeddingNet(BaseEmbeddingNet):
    def __init__(self, embedding_dim: int, num_blocks: int):
        super().__init__(embedding_dim, num_blocks, ResBlock(64))


class SEResEmbeddingNet(BaseEmbeddingNet):
    def __init__(self, embedding_dim: int, num_blocks: int, reduction: int = 8):
        super().__init__(embedding_dim, num_blocks, SEResBlock(64, reduction))


class MultiTaskEmbeddingNet(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_blocks: int,
        block: nn.Module,
    ):
        super().__init__()
        self.conv_input = nn.Conv2d(17, 64, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(64)
        self.res_layers = nn.Sequential(*[block for _ in range(num_blocks)])
        self.conv_output = nn.Conv2d(64, 32, 1)
        self.flat_dim = 32 * 8 * 8
        self.embedding_head = nn.Linear(32 * 8 * 8, embedding_dim)
        self.puzzle_head = nn.Sequential(
            nn.Linear(self.flat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor):
        x = F.relu(self.bn_input(self.conv_input(x)))
        x = self.res_layers(x)
        x = F.relu(self.conv_output(x))
        features = x.view(x.size(0), -1)
        emb = self.embedding_head(features)
        emb = F.normalize(emb, p=2, dim=1)
        puzzle_logits = self.puzzle_head(features)

        return emb, puzzle_logits


class MultiTaskSEResEmbeddingNet(MultiTaskEmbeddingNet):
    def __init__(self, embedding_dim: int, num_blocks: int, reduction: int = 8):
        super().__init__(embedding_dim, num_blocks, SEResBlock(64, reduction))
