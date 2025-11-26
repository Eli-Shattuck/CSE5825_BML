import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import your existing components
from beschess.components.net.resnet import SEResEmbeddingNet
from beschess.components.utils import (
    CheckpointManager,
    compute_tsne_embeddings,
    plot_tsne_embeddings,
)
from beschess.utils import packed_to_tensor
from beschess.data.embedding import (
    BalancedBatchSampler,
    DirectLoader,
    PuzzleDataset,
    generate_split_indices,
)

# --- Configuration ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

EPOCHS = 50
LR = 1e-3
EMBEDDING_DIM = 128
BATCH_SIZE = 4096
NUM_CLASSES = 16

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "processed"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "checkpoints"
LOG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "logs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Loading ---
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

train_loader = DirectLoader(
    dataset,
    BalancedBatchSampler(
        dataset,
        q_train,
        p_train,
        batch_size=BATCH_SIZE,
        steps_per_epoch=1000,
    ),
    device=device,
)

# Validation Loader
val_indices = np.concatenate([q_val, p_val]).tolist()
# Evaluate on a smaller subset to save time, but large enough to be representative
np.random.shuffle(val_indices)
val_subset = Subset(dataset, val_indices[:10000])
val_loader = DataLoader(
    val_subset, batch_size=1024, shuffle=False, num_workers=0
)  # num_workers=0 for safety with DirectLoader logic if mixed

# --- Model Setup ---
model = SEResEmbeddingNet(
    embedding_dim=EMBEDDING_DIM,
    num_blocks=10,
    reduction=16,
).to(device)

# CLASSIFICATION HEAD: Maps 128-dim embedding to 16 Class Logits
classifier_head = nn.Linear(EMBEDDING_DIM, NUM_CLASSES).to(device)

# LOSS FUNCTION: Multi-label classification
# We weight positive examples higher because 'Quiet' is common, but specific tags are rare.
pos_weight = torch.ones(NUM_CLASSES).to(device) * 5.0
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# OPTIMIZER: Optimize both Backbone and Head
optimizer = optim.AdamW(
    list(model.parameters()) + list(classifier_head.parameters()),
    lr=LR,
    weight_decay=1e-4,
)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    pct_start=0.1,
)

# Logging
run_name = f"BCE_ResNet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
(CHECKPOINT_DIR / run_name).mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=LOG_DIR / run_name)
global_step = 0

print("Starting Training (BCE Classification Strategy)...")

# --- Training Loop ---
for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    model.train()
    classifier_head.train()
    total_train_loss = 0.0

    pbar = tqdm(train_loader, total=len(train_loader), desc="Batches", leave=False)

    for i, (inputs, targets) in enumerate(pbar):
        # inputs: (B, 17, 8, 8)
        # targets: (B, 16) [Multi-hot Float]

        optimizer.zero_grad()

        # 1. Get Embeddings (Unnormalized)
        embeddings = model(inputs)

        # 2. Project to Class Logits
        logits = classifier_head(embeddings)

        # 3. Calculate Loss
        loss = loss_fn(logits, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()

        if global_step % 50 == 0:
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], global_step)

        global_step += 1

    avg_train_loss = total_train_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    classifier_head.eval()
    total_val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs_cpu, targets_cpu in val_loader:
            inputs = inputs_cpu.to(device)
            targets = targets_cpu.to(device)

            emb = model(inputs)
            logits = classifier_head(emb)
            val_loss = loss_fn(logits, targets)

            total_val_loss += val_loss.item()

            # Metrics: Sigmoid -> Threshold 0.5
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)

    # Compute Simple Accuracy Metrics
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Exact Match Ratio (Subset Accuracy)
    exact_matches = np.all(all_preds == all_targets, axis=1)
    subset_acc = np.mean(exact_matches)

    # Hamming Accuracy (Element-wise)
    hamming_acc = np.mean(all_preds == all_targets)

    writer.add_scalar("Val/Loss", avg_val_loss, global_step)
    writer.add_scalar("Val/Subset_Accuracy", subset_acc, global_step)
    writer.add_scalar("Val/Hamming_Accuracy", hamming_acc, global_step)

    print(
        f"Epoch {epoch + 1} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Subset Acc: {subset_acc:.4f}"
    )

    # Checkpoint
    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "head_state_dict": classifier_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            CHECKPOINT_DIR / run_name / f"checkpoint_{epoch}.pt",
        )

writer.close()
