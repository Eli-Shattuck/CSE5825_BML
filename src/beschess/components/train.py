import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

# Import cross entropy loss
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from beschess.components.loss import ProxyAnchor
from beschess.components.net.resnet import SEResEmbeddingNet
from beschess.components.utils import (
    CheckpointManager,
    compute_proxy_hitrate,
    compute_proxy_map,
    compute_quiet_margin,
    compute_tsne_embeddings,
    plot_tsne_embeddings,
    evaluate_proxy_cos,
)
from beschess.data.embedding import (
    BalancedBatchSampler,
    DirectLoader,
    PuzzleDataset,
    generate_split_indices,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

EPOCHS = 50
MODEL_LR = 1e-4
LOSS_LR = 1e-2
EMBEDDING_DIM = 128
BATCH_SIZE = 4096

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "processed"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "checkpoints"
LOG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "logs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# # Sanity check dataset
# splits = generate_split_indices(dataset, test_split=0.001, val_split=0.001)
# q_train, p_train = splits["train"]
# q_val, p_val = splits["val"]
# q_test, p_test = splits["test"]
#
# train_loader = DataLoader(
#     dataset,
#     batch_sampler=BalancedBatchSampler(
#         dataset,
#         q_val,
#         p_val,
#         batch_size=BATCH_SIZE,
#         steps_per_epoch=10,
#     ),
#     num_workers=4,
# )
#
# VAL_BATCH_SIZE = 512
# VAL_STEPS = (len(q_val) + len(p_val)) // VAL_BATCH_SIZE + 1
# val_loader = DataLoader(
#     dataset,
#     batch_sampler=BalancedBatchSampler(
#         dataset,
#         q_val,
#         p_val,
#         batch_size=VAL_BATCH_SIZE,
#         steps_per_epoch=VAL_STEPS,
#     ),
#     num_workers=4,
# )

splits = generate_split_indices(dataset)
q_train, p_train = splits["train"]
q_val, p_val = splits["val"]
q_test, p_test = splits["test"]

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

val_loader = DataLoader(
    dataset,
    batch_sampler=BalancedBatchSampler(
        dataset,
        q_val,
        p_val,
        batch_size=512,
    ),
    num_workers=4,
)

model = SEResEmbeddingNet(
    embedding_dim=EMBEDDING_DIM,
    num_blocks=10,
).to(device)

loss_fn = ProxyAnchor(
    n_classes=16,
    embedding_dim=EMBEDDING_DIM,
    margin=0.1,
    alpha=32,
).to(device)

optimizer = torch.optim.AdamW(
    [
        {"params": model.parameters(), "lr": MODEL_LR, "weight_decay": 1e-4},
        {"params": loss_fn.parameters(), "lr": LOSS_LR, "weight_decay": 0},
    ]
)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[MODEL_LR, LOSS_LR],
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    pct_start=0.1,
)

run_name = f"{model.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
(CHECKPOINT_DIR / run_name).mkdir(parents=True, exist_ok=True)
checkpoint_manager = CheckpointManager(
    CHECKPOINT_DIR / run_name,
    metric_key="val_map@3",
)

writer = SummaryWriter(log_dir=LOG_DIR / run_name)
global_step = 0

for epoch in tqdm(range(EPOCHS), desc="Training Epochs"):
    model.train()
    loss_fn.train()
    total_train_loss = 0.0

    pbar = tqdm(
        train_loader, total=len(train_loader), desc="Training Batches", leave=False
    )

    for i, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        embeddings = model(inputs)
        batch_loss = loss_fn(embeddings, targets)
        batch_loss.backward()
        optimizer.step()
        scheduler.step()

        total_train_loss += batch_loss.item()

        if global_step % 50 == 0:
            writer.add_scalar("Train/Loss", batch_loss.item(), global_step)

            lrs = scheduler.get_last_lr()
            writer.add_scalar("Train/LR_Backbone", lrs[0], global_step)
            writer.add_scalar("Train/LR_Proxy", lrs[1], global_step)

            with torch.no_grad():
                proxy_norm = torch.norm(loss_fn.proxies, dim=1).mean()
                writer.add_scalar("Debug/Proxy_Norms", proxy_norm, global_step)

        global_step += 1

    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_val_loss = 0.0

    similarity_matrix, val_labels = evaluate_proxy_cos(
        model, loss_fn, val_loader, device
    )

    similarity_matrix = similarity_matrix.cpu()
    val_labels = val_labels.cpu()

    k_list = [1, 3]
    max_k = max(k_list)
    _, top_indices = torch.topk(similarity_matrix, k=max_k, dim=1)

    hitrate = compute_proxy_hitrate(top_indices, val_labels, k_list)
    val_map = compute_proxy_map(top_indices, val_labels, k_list)
    avg_margin, margin_acc = compute_quiet_margin(similarity_matrix, val_labels)

    avg_val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            embeddings = model(inputs)
            batch_loss = loss_fn(embeddings, targets)
            avg_val_loss += batch_loss.item()

    avg_val_loss /= len(val_loader)

    metrics = {
        "val_map@1": val_map[1],
        "val_map@3": val_map[3],
        "val_hitrate@1": hitrate[1],
        "val_hitrate@3": hitrate[3],
        "val_quiet_margin": avg_margin,
        "val_quiet_margin_acc": margin_acc,
        "train_loss": avg_train_loss,
    }

    writer.add_scalar("Val/MAP@1", val_map[1], global_step)
    writer.add_scalar("Val/MAP@3", val_map[3], global_step)
    writer.add_scalar("Val/HitRate@1", hitrate[1], global_step)
    writer.add_scalar("Val/HitRate@3", hitrate[3], global_step)
    writer.add_scalar("Val/Quiet_Margin", avg_margin, global_step)
    writer.add_scalar("Val/Quiet_Acc", margin_acc, global_step)

    writer.add_scalar("Loss/Epoch_Avg", avg_train_loss, global_step)
    writer.add_scalar("Loss/Epoch_Avg", avg_val_loss, global_step)

    checkpoint_manager.check(
        model,
        loss_fn,
        optimizer,
        scheduler,
        metrics,
        epoch,
    )

    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        board_embeddings, board_labels, proxy_embeddings, proxy_labels = (
            compute_tsne_embeddings(
                model,
                loss_fn,
                val_loader,
                device,
            )
        )
        fig = plot_tsne_embeddings(
            board_embeddings,
            board_labels,
            proxy_embeddings,
            title=f"Epoch {epoch + 1} Embeddings",
        )
        writer.add_figure("Embeddings/TSNE", fig, global_step)
        plt.close(fig)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"MAP@3: {val_map[3]:.4f} | "
        f"HR@3: {hitrate[3]:.4f} | "
        f"Margin: {avg_margin:.4f}"
    )

writer.close()
