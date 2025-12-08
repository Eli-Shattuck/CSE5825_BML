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
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm

from beschess.components.loss import ProxyAnchor
from beschess.components.net.vit import MultiTaskViT, MultiTaskViT2D
from beschess.components.utils import (
    CheckpointManager,
    compute_proxy_hitrate,
    compute_proxy_map,
    compute_tsne_embeddings,
    compute_binary_accuracy,
    compute_geometry_metrics,
    evaluate_proxy_cos,
    plot_tsne_embeddings,
)
from beschess.data.embedding import (
    BalancedBatchSampler,
    DirectLoader,
    PuzzleDataset,
    generate_split_indices,
)

TAG_NAMES = [
    "MatingNet",
    "SpecialMove",
    "Promotion",
    "DoubleAttack",
    "LinearAttack",
    "Punishment",
    "ForcingMove",
]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

GRAD_CLIP = 1.0
EPOCHS = 50
MODEL_LR = 1e-4 * 0.5
LOSS_LR = 1e-2
EMBEDDING_DIM = 128
BATCH_SIZE = 4096
LAMBDA_BCE = 5.0

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

# Use High Precision for Matrix Multiplication on Ampere+ GPUs
torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

quiet_boards_file = DATA_DIR / "quiet_boards_preeval.npy"
hard_negatives_file = DATA_DIR / "hard_negatives.npy"
puzzle_boards_file = DATA_DIR / "boards_packed.npy"
puzzle_labels_file = DATA_DIR / "tags_packed.npy"

quiet_boards = np.load(quiet_boards_file)
puzzle_boards = np.load(puzzle_boards_file)
puzzle_labels = np.load(puzzle_labels_file)

try:
    hard_negatives = np.load(hard_negatives_file)
    print(f"Found {len(hard_negatives)} Hard Negatives.")

    n_hard = len(hard_negatives)
    n_quiet = len(quiet_boards)

    if n_quiet > n_hard:
        print(
            f"Balancing: Downsampling Quiet Boards ({n_quiet}) to match Hard Negatives ({n_hard})..."
        )
        indices = np.random.choice(n_quiet, n_hard, replace=False)
        quiet_subset = quiet_boards[indices]
        negatives_combined = np.concatenate([quiet_subset, hard_negatives], axis=0)
    else:
        print(
            f"Balancing: Using all available boards (Quiet: {n_quiet}, Hard: {n_hard})."
        )
        negatives_combined = np.concatenate([quiet_boards, hard_negatives], axis=0)

except FileNotFoundError:
    print("WARNING: hard_negatives.npy not found! Training on quiet boards only.")
    negatives_combined = quiet_boards

dataset = PuzzleDataset(
    quiet_boards=negatives_combined,
    puzzle_boards=puzzle_boards,
    puzzle_labels=puzzle_labels,
)

splits = generate_split_indices(dataset)
q_train, p_train = splits["train"]
q_val, p_val = splits["val"]
q_test, p_test = splits["test"]

with open(CHECKPOINT_DIR / "test_indices.txt", "w") as f:
    for q_idx in q_test:
        f.write(f"q,{q_idx}\n")
    for p_idx in p_test:
        f.write(f"p,{p_idx}\n")

train_loader = DataLoader(
    dataset,
    batch_sampler=BalancedBatchSampler(
        dataset,
        q_train,
        p_train,
        batch_size=BATCH_SIZE,
        steps_per_epoch=2000,
    ),
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

VAL_BATCH_SIZE = 512
VAL_STEPS = (len(q_val) + len(p_val)) // VAL_BATCH_SIZE + 1
val_loader = DataLoader(
    dataset,
    batch_sampler=BalancedBatchSampler(
        dataset,
        q_val,
        p_val,
        batch_size=VAL_BATCH_SIZE,
        steps_per_epoch=VAL_STEPS,
    ),
    num_workers=4,
    pin_memory=True,
)

val_puzzle_subset = Subset(dataset, p_val)
val_puzzle_loader = DataLoader(
    val_puzzle_subset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

model = MultiTaskViT(
    in_channels=17,
    embed_dim=256,
    num_heads=8,
    depth=6,
    out_dim=EMBEDDING_DIM,
).to(device)

model = torch.compile(model, mode="max-autotune")

loss_fn_emb = ProxyAnchor(
    n_classes=len(TAG_NAMES),
    embedding_dim=EMBEDDING_DIM,
    margin=0.4,
    alpha=32,
).to(device)

loss_fn_binary = nn.BCEWithLogitsLoss().to(device)

optimizer = torch.optim.AdamW(
    [
        {"params": model.parameters(), "lr": MODEL_LR, "weight_decay": 1e-4},
        {"params": loss_fn_emb.parameters(), "lr": LOSS_LR, "weight_decay": 0},
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
    save_interval=5,
)

scaler = GradScaler(enabled=False)

writer = SummaryWriter(log_dir=LOG_DIR / run_name)
global_step = 0

for epoch in tqdm(range(EPOCHS), desc="Training Epochs"):
    model.train()
    loss_fn_emb.train()
    total_train_loss = 0.0
    total_binary_acc = 0.0

    pbar = tqdm(
        train_loader, total=len(train_loader), desc="Training Batches", leave=False
    )

    for i, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        is_puzzle_mask = targets[:, 0] == 0
        puzzle_inputs = inputs[is_puzzle_mask]
        puzzle_targets = targets[is_puzzle_mask][:, 1:]

        binary_targets = is_puzzle_mask.float().unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            embeddings, puzzle_logits = model(inputs)

            loss_emd = loss_fn_emb(embeddings[is_puzzle_mask], puzzle_targets)

            loss_bce = loss_fn_binary(puzzle_logits, binary_targets)

            batch_loss_emd = loss_emd + (LAMBDA_BCE * loss_bce)

        scaler.scale(batch_loss_emd).backward()

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_train_loss += batch_loss_emd.detach()

        with torch.no_grad():
            preds = (torch.sigmoid(puzzle_logits) > 0.5).float()
            acc = (preds == binary_targets).float().mean()
            total_binary_acc += acc.item()

        if global_step % 50 == 0:
            writer.add_scalar("Train/Loss_Total", batch_loss_emd.item(), global_step)
            writer.add_scalar("Train/Loss_Embedding", loss_emd.item(), global_step)
            writer.add_scalar("Train/Loss_BCE", loss_bce.item(), global_step)
            writer.add_scalar("Train/Binary_Acc", acc.item(), global_step)

            lrs = scheduler.get_last_lr()
            writer.add_scalar("Train/LR_Backbone", lrs[0], global_step)
            writer.add_scalar("Train/LR_Proxy", lrs[1], global_step)

            # Debugging Proxy Collapse
            with torch.no_grad():
                if hasattr(loss_fn_emb, "proxies"):
                    proxy_norm = torch.norm(loss_fn_emb.proxies, dim=1).mean()
                    writer.add_scalar("Debug/Proxy_Norms", proxy_norm, global_step)

        global_step += 1

    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()

    similarity_matrix, val_labels = evaluate_proxy_cos(
        model, loss_fn_emb, val_puzzle_loader, device
    )

    similarity_matrix = similarity_matrix.cpu()
    val_labels = val_labels.cpu()[:, 1:]

    k_list = [1, 3]
    max_k = max(k_list)
    _, top_indices = torch.topk(similarity_matrix, k=max_k, dim=1)

    hitrate = compute_proxy_hitrate(top_indices, val_labels, k_list)
    val_map = compute_proxy_map(top_indices, val_labels, k_list)

    val_binary_acc = compute_binary_accuracy(model, val_loader, device)

    avg_pos_dist, avg_neg_dist, avg_proxy_spread = compute_geometry_metrics(
        model,
        loss_fn_emb,
        val_puzzle_loader,
        device,
    )

    metrics = {
        "val_map@1": val_map[1],
        "val_map@3": val_map[3],
        "val_hitrate@1": hitrate[1],
        "val_hitrate@3": hitrate[3],
        "val_binary_acc": val_binary_acc,
        "train_loss": avg_train_loss,
        "avg_pos_dist": avg_pos_dist,
        "avg_neg_dist": avg_neg_dist,
        "avg_proxy_spread": avg_proxy_spread,
    }

    writer.add_scalar("Val/MAP@1", val_map[1], global_step)
    writer.add_scalar("Val/MAP@3", val_map[3], global_step)
    writer.add_scalar("Val/HitRate@1", hitrate[1], global_step)
    writer.add_scalar("Val/HitRate@3", hitrate[3], global_step)
    writer.add_scalar("Val/Binary_Acc", val_binary_acc, global_step)
    writer.add_scalar("Val/Avg_Pos_Dist", avg_pos_dist, global_step)
    writer.add_scalar("Val/Avg_Neg_Dist", avg_neg_dist, global_step)
    writer.add_scalar("Val/Avg_Proxy_Spread", avg_proxy_spread, global_step)

    checkpoint_manager.check(
        model,
        loss_fn_emb,
        optimizer,
        scheduler,
        metrics,
        epoch,
    )

    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        print("Generating Final T-SNE...")
        board_embeddings, board_labels, proxy_embeddings, proxy_labels, all_probs = (
            compute_tsne_embeddings(
                model,
                loss_fn_emb,
                val_loader,
                device,
            )
        )
        fig = plot_tsne_embeddings(
            board_embeddings,
            board_labels,
            proxy_embeddings,
            all_probs,
            title=f"Epoch {epoch + 1} Embeddings",
            tag_names=TAG_NAMES,
        )
        writer.add_figure("Embeddings/TSNE", fig, global_step)
        plt.close(fig)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"MAP@3: {val_map[3]:.4f} | "
        f"HR@1: {hitrate[1]:.4f} | "
        f"Pos/Neg Dist: {avg_pos_dist:.2f}/{avg_neg_dist:.2f}"
    )

writer.close()
