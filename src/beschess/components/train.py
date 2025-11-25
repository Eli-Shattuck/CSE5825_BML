import random
from pathlib import Path

import chess
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from beschess.components.loss import ProxyAnchor
from beschess.components.net.resnet import ResEmbeddingNet, SEResEmbeddingNet
from beschess.data.embedding import (
    BalancedBatchSampler,
    DirectLoader,
    PuzzleDataset,
    generate_split_indices,
)
from beschess.utils import tensor_to_board

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

EPOCHS = 50
MODEL_LR = 1e-4
LOSS_LR = 1e-3
EMBEDDING_DIM = 128

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "processed"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

quiet_boards_file = DATA_DIR / "quiet_boards_preeval.npy"
puzzle_boards_file = DATA_DIR / "boards_packed.npy"
puzzle_labels_file = DATA_DIR / "tags_packed.npy"

# quiet_boards = np.load(quiet_boards_file, mmap_mode="r")
# puzzle_boards = np.load(puzzle_boards_file, mmap_mode="r")
# puzzle_labels = np.load(puzzle_labels_file, mmap_mode="r")

# subset for quick testing
quiet_boards = np.load(quiet_boards_file, mmap_mode="r")[:100000]
puzzle_boards = np.load(puzzle_boards_file, mmap_mode="r")[:100000]
puzzle_labels = np.load(puzzle_labels_file, mmap_mode="r")[:100000]

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
        batch_size=2048,
        steps_per_epoch=2000,
    ),
    device=device,
)

val_indices = np.concat([q_val, p_val]).tolist()
vaL_subset = Subset(dataset, val_indices)
val_loader = DataLoader(
    vaL_subset,
    batch_size=512,
    shuffle=False,
    num_workers=4,
)

model = SEResEmbeddingNet(
    embedding_dim=EMBEDDING_DIM,
    num_blocks=10,
    reduction=16,
).to(device)

loss = ProxyAnchor(
    n_classes=16,
    embedding_dim=EMBEDDING_DIM,
    margin=0.1,
    alpha=32,
).to(device)

optimizer = torch.optim.AdamW(
    [
        {"params": model.parameters(), "lr": MODEL_LR, "weight_decay": 1e-5},
        {"params": loss.parameters(), "lr": LOSS_LR, "weight_decay": 0},
    ]
)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[MODEL_LR, LOSS_LR],
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    pct_start=0.3,
)

for epoch in tqdm(range(EPOCHS), desc="Training Epochs"):
    model.train()
    total_train_loss = 0.0

    # for batch_idx, (inputs, targets) in enumerate(train_loader):
    for batch_idx, (inputs, targets) in tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc="Training Batches",
        leave=False,
    ):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        embeddings = model(inputs)
        batch_loss = loss(embeddings, targets)
        batch_loss.backward()
        optimizer.step()
        scheduler.step()

        total_train_loss += batch_loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            embeddings = model(inputs)
            batch_loss = loss(embeddings, targets)

            total_val_loss += batch_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] - "
        f"Train Loss: {avg_train_loss:.4f} - "
        f"Val Loss: {avg_val_loss:.4f}"
    )
    break  # Remove this line to run full training
