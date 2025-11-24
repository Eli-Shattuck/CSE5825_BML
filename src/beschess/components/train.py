import random
from pathlib import Path

import chess
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from beschess.data.embedding import (
    BalancedBatchSampler,
    PuzzleDataset,
    DirectLoader,
    generate_split_indices,
)
from beschess.components.net.resnet import SEResEmbeddingNet, ResEmbeddingNet
from beschess.components.loss import ProxyAnchor
from beschess.utils import tensor_to_board

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "processed"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

quiet_boards_file = DATA_DIR / "quiet_boards_preeval.npy"
puzzle_boards_file = DATA_DIR / "boards_packed.npy"
puzzle_labels_file = DATA_DIR / "tags_packed.npy"

quiet_boards = np.load(quiet_boards_file, mmap_mode="r")
puzzle_boards = np.load(puzzle_boards_file, mmap_mode="r")
puzzle_labels = np.load(puzzle_labels_file, mmap_mode="r")

dataset = PuzzleDataset(
    quiet_boards=quiet_boards,
    puzzle_boards=puzzle_boards,
    puzzle_labels=puzzle_labels,
)

splits = generate_split_indices(dataset)
q_train, p_train = splits["train"]
q_val, p_val = splits["val"]
q_test, p_test = splits["test"]


train_loader = DirectLoader(
    dataset,
    BalancedBatchSampler(
        dataset,
        q_val,
        p_val,
        batch_size=32,
        steps_per_epoch=1000,
    ),
    device=device,
)

val_indices = np.concatenate([q_val, p_val])
vaL_subset = Subset(dataset, val_indices)
val_loader = DataLoader(
    vaL_subset,
    batch_size=512,
    shuffle=False,
    num_workers=4,
)

model = SEResEmbeddingNet(
    embedding_dim=128,
    num_blocks=10,
    reduction=16,
).to(device)

loss = ProxyAnchor(
    n_classes=16,
    embedding_dim=128,
    margin=0.1,
    alpha=32,
).to(device)

optimizer = torch.optim.AdamW(
    [
        {"params": model.parameters(), "lr": 1e-4, "weight_decay": 1e-5},
        {"params": loss.parameters(), "lr": 1e-3, "weight_decay": 0},
    ]
)
