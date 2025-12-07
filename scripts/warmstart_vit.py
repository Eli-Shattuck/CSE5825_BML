import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# --- BESCHESS IMPORTS ---
# Ensure these match your package structure
from beschess.components.loss import ProxyAnchor
from beschess.components.net.vit import MultiTaskViT
from beschess.components.utils import (
    CheckpointManager,
    clean_state_dict,
    compute_proxy_hitrate,
    compute_proxy_map,
    compute_tsne_embeddings,
    compute_binary_accuracy,
    evaluate_proxy_cos,
    plot_tsne_embeddings,
)
from beschess.data.embedding import (
    BalancedBatchSampler,
    PuzzleDataset,
    generate_split_indices,
)

# ==========================================
# CONFIGURATION
# ==========================================

# 1. NEW 7-CLASS MAPPING
TAG_NAMES = [
    "MatingNet",
    "SpecialMove",
    "Promotion",
    "DoubleAttack",
    "LinearAttack",
    "Punishment",
    "ForcingMove",
]

# 2. PATHS
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

# !!! POINT THIS TO YOUR BEST PREVIOUS MODEL !!!
PRETRAINED_CHECKPOINT = (
    CHECKPOINT_DIR / "OptimizedModule_20251203_151336" / "best_checkpoint.pth"
)

# 3. HYPERPARAMETERS
SEED = 42
BATCH_SIZE = 4096
EMBEDDING_DIM = 128
LAMBDA_BCE = 5.0
GRAD_CLIP = 1.0

# Stage 1: Warmup Heads (Fast, high LR, Backbone Frozen)
WARMUP_EPOCHS = 1
WARMUP_LR = 1e-3

# Stage 2: Fine-Tuning (Slow, low LR, Backbone Unfrozen)
FINETUNE_EPOCHS = 10
FINETUNE_LR_BACKBONE = 1e-4  # Low to preserve "Chess Grammar"
FINETUNE_LR_HEADS = 1e-3

# ==========================================
# SETUP
# ==========================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. DATA LOADING & BALANCING
# ==========================================
print("Loading data...")
quiet_boards_file = DATA_DIR / "quiet_boards_preeval.npy"
hard_negatives_file = DATA_DIR / "hard_negatives.npy"
puzzle_boards_file = DATA_DIR / "boards_packed.npy"
puzzle_labels_file = DATA_DIR / "tags_packed.npy"  # Ensure this uses NEW 7-class map

quiet_boards = np.load(quiet_boards_file)
puzzle_boards = np.load(puzzle_boards_file)
puzzle_labels = np.load(puzzle_labels_file)

try:
    hard_negatives = np.load(hard_negatives_file)
    print(f"Found {len(hard_negatives)} Hard Negatives.")

    # --- BALANCING STRATEGY ---
    # We want roughly 50% Quiet / 50% Hard Negatives in the "Non-Puzzle" pool.
    # We downsample the larger set to match the smaller set.
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

print(f"Total Negatives (Balanced): {len(negatives_combined)}")
print(f"Total Puzzles: {len(puzzle_boards)}")

dataset = PuzzleDataset(
    quiet_boards=negatives_combined,
    puzzle_boards=puzzle_boards,
    puzzle_labels=puzzle_labels,
)

splits = generate_split_indices(dataset)
q_train, p_train = splits["train"]
q_val, p_val = splits["val"]

# ==========================================
# 2. MODEL SURGERY
# ==========================================
print("Initializing Model...")
model = MultiTaskViT(
    in_channels=17,
    embed_dim=256,
    num_heads=8,
    depth=6,
    out_dim=EMBEDDING_DIM,
).to(device)

if PRETRAINED_CHECKPOINT.exists():
    print(f"Loading Backbone from {PRETRAINED_CHECKPOINT}...")
    checkpoint = torch.load(PRETRAINED_CHECKPOINT, map_location=device)

    # Handle state dict structure (sometimes wrapped in 'model_state_dict')
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = clean_state_dict(state_dict)
    model_state = model.state_dict()

    # --- SURGERY ---
    # 1. Keep Backbone + Metric Head (Projection)
    # 2. Drop Classifier Head (Needs to relearn Hard Negatives)
    filtered_state = {
        k: v
        for k, v in state_dict.items()
        if k in model_state
        and "classifier_head" not in k
        and v.shape == model_state[k].shape
    }

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    print(
        f"Weights loaded. Re-initialized layers: {len(model_state) - len(filtered_state)}"
    )
else:
    raise FileNotFoundError(f"Checkpoint not found at {PRETRAINED_CHECKPOINT}")

# Explicitly reset classifier head weights (just to be safe)
print("Resetting Classifier Head weights...")
for layer in model.classifier_head.children():
    if hasattr(layer, "reset_parameters"):
        layer.reset_parameters()

# Initialize Proxy Loss with NEW Class Count (7)
loss_fn_emb = ProxyAnchor(
    n_classes=len(TAG_NAMES),
    embedding_dim=EMBEDDING_DIM,
    margin=0.4,
    alpha=8,
).to(device)

loss_fn_binary = nn.BCEWithLogitsLoss().to(device)

# ==========================================
# 3. TRAINING INFRASTRUCTURE
# ==========================================
scaler = GradScaler()
run_name = f"FineTune_7Class_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
(CHECKPOINT_DIR / run_name).mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=LOG_DIR / run_name)
checkpoint_manager = CheckpointManager(
    CHECKPOINT_DIR / run_name,
    metric_key="val_map@3",
    save_interval=1,  # Save best model frequently during fine-tuning
)

# Dataloaders
train_loader = DataLoader(
    dataset,
    batch_sampler=BalancedBatchSampler(
        dataset, q_train, p_train, batch_size=BATCH_SIZE, steps_per_epoch=2000
    ),
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

val_puzzle_loader = DataLoader(Subset(dataset, p_val), batch_size=512, num_workers=4)
# Full val loader for T-SNE
val_loader = DataLoader(
    dataset,
    batch_sampler=BalancedBatchSampler(
        dataset, q_val, p_val, batch_size=512, steps_per_epoch=100
    ),
    num_workers=4,
)

global_step = 0


def run_epoch(optimizer, scheduler=None, desc="Training"):
    global global_step
    model.train()
    loss_fn_emb.train()

    total_loss = 0.0
    total_binary_acc = 0.0  # <--- NEW: Track accumulator

    pbar = tqdm(train_loader, desc=desc, leave=False)

    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Identify Puzzles vs Negatives
        is_puzzle_mask = targets[:, 0] == 0
        puzzle_inputs = inputs[is_puzzle_mask]
        puzzle_targets = targets[is_puzzle_mask][:, 1:]

        optimizer.zero_grad()

        # Forward
        embeddings, puzzle_logits = model(inputs)

        # 1. Metric Loss
        if puzzle_inputs.size(0) > 0:
            puzzle_embeddings = embeddings[is_puzzle_mask]
            loss_emd = loss_fn_emb(puzzle_embeddings, puzzle_targets)
        else:
            loss_emd = torch.tensor(0.0, device=device)

        # 2. Binary Loss
        # Ensure target shape matches logits (B, 1)
        binary_targets = is_puzzle_mask.float().unsqueeze(1)
        loss_bce = loss_fn_binary(puzzle_logits, binary_targets)

        # Combine
        total = loss_emd + (LAMBDA_BCE * loss_bce)

        # Backward
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        if scheduler:
            scheduler.step()

        # --- NEW: CALCULATE ACCURACY ---
        # 1. Sigmoid to get probability (0.0 to 1.0)
        # 2. > 0.5 to get class (True/False)
        # 3. Float to get (1.0/0.0)
        preds = (torch.sigmoid(puzzle_logits) > 0.5).float()

        # Compare prediction to target
        batch_acc = (preds == binary_targets).float().mean()
        total_binary_acc += batch_acc.item()
        # -------------------------------

        total_loss += total.item()

        # Logging
        if global_step % 50 == 0:
            writer.add_scalar("Train/Loss_Total", total.item(), global_step)
            writer.add_scalar("Train/Loss_Proxy", loss_emd.item(), global_step)
            writer.add_scalar("Train/Loss_Binary", loss_bce.item(), global_step)

            # <--- NEW: Log Accuracy
            writer.add_scalar("Train/Binary_Acc", batch_acc.item(), global_step)

            # Optional: Log current LR
            if scheduler:
                writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], global_step)

            pbar.set_postfix(
                {"Loss": f"{total.item():.4f}", "Acc": f"{batch_acc.item():.4f}"}
            )

        global_step += 1

        # Update progress bar text

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_binary_acc / len(train_loader)

    # Print epoch summary
    print(f"{desc} - Avg Loss: {avg_loss:.4f} | Avg Binary Acc: {avg_acc:.4f}")

    return avg_loss


# ==========================================
# STAGE 1: HEAD WARMUP (Backbone Frozen)
# ==========================================
print("\n=== STAGE 1: HEAD WARMUP (Backbone Frozen) ===")

# Freeze Backbone and Metric Projection
for name, param in model.named_parameters():
    # We only train classifier_head (new task) and loss_fn_emb (new proxies)
    if "classifier_head" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Optimizer for Stage 1
optimizer_s1 = optim.AdamW(
    [
        {
            "params": filter(lambda p: p.requires_grad, model.parameters()),
            "lr": WARMUP_LR,
        },
        {
            "params": loss_fn_emb.parameters(),
            "lr": WARMUP_LR * 10,
        },  # Proxies need high LR to move initially
    ]
)

for epoch in range(WARMUP_EPOCHS):
    loss = run_epoch(optimizer_s1, None, f"Warmup Epoch {epoch + 1}")
    print(f"Warmup Epoch {epoch + 1} Loss: {loss:.4f}")

    checkpoint_manager._save_checkpoint(
        model,
        loss_fn_emb,
        optimizer_s1,
        None,
        {"train_loss": loss},
        f"warmup_epoch_{epoch + 1}",
    )

# ==========================================
# STAGE 2: FINE-TUNING (Backbone Unfrozen)
# ==========================================

# print("\n=== STAGE 2: FULL FINE-TUNING (Backbone Unfrozen) ===")
#
# # Unfreeze Everything
# for param in model.parameters():
#     param.requires_grad = True
#
# # Compile model
# torch._dynamo.reset()
# print("Compiling model...")
# model = torch.compile(model, mode="reduce-overhead", dynamic=True)
#
# # AGGRESSIVE OPTIMIZER SETTINGS
# # 1. Higher LR for Backbone (1e-4, matching heads)
# # 2. Lower Weight Decay (1e-4, preventing over-regularization)
# optimizer_s2 = optim.AdamW(
#     [
#         {"params": model.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
#         {"params": loss_fn_emb.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
#     ]
# )
#
# # REMOVED SCHEDULER
# # We want constant, raw power to break the stagnation.
# scheduler_s2 = None
#
# for epoch in range(FINETUNE_EPOCHS):
#     # Pass 'None' for scheduler
#     train_loss = run_epoch(
#         optimizer_s2, None, f"FineTune Epoch {epoch + 1}/{FINETUNE_EPOCHS}"
#     )
#
#     # --- DEBUGGING: Check if weights are actually moving ---
#     # Print the norm of the first layer's weights. If this doesn't change, we are frozen.
#     with torch.no_grad():
#         param_norm = model.patch_proj.weight.norm().item()
#         grad_scale = scaler.get_scale()
#         print(f"DEBUG: Layer Norm: {param_norm:.5f} | Grad Scaler: {grad_scale}")
#
#     # --- EVALUATION ---
#     model.eval()
#
#     # 1. Compute Metric Performance (MAP/HitRate)
#     similarity_matrix, val_labels = evaluate_proxy_cos(
#         model, loss_fn_emb, val_puzzle_loader, device
#     )
#
#     similarity_matrix = similarity_matrix.cpu()
#     val_labels = val_labels.cpu()[:, 1:]  # Remove binary index
#
#     _, top_indices = torch.topk(similarity_matrix, k=3, dim=1)
#     k_list = [1, 3]
#     val_map = compute_proxy_map(top_indices, val_labels, k_list)
#     hitrate = compute_proxy_hitrate(top_indices, val_labels, k_list)
#     val_binary_acc = compute_binary_accuracy(model, val_loader, device)
#
#     print(
#         f"Epoch {epoch + 1} | Loss: {train_loss:.4f} | MAP@3: {val_map[3]:.4f} | HR@1: {hitrate[1]:.4f} | Binary Acc: {val_binary_acc:.4f}"
#     )
#
#     writer.add_scalar("Val/MAP@3", val_map[3], global_step)
#     writer.add_scalar("Val/HitRate@1", hitrate[1], global_step)
#     writer.add_scalar("Val/Binary_Acc", val_binary_acc, global_step)
#
#     metrics = {
#         "binary_acc": val_binary_acc,
#         "val_map@3": val_map[3],
#         "val_hitrate@1": hitrate[1],
#         "train_loss": train_loss,
#     }
#
#     checkpoint_manager.check(
#         model, loss_fn_emb, optimizer_s2, scheduler_s2, metrics, epoch
#     )
#
#     # 2. T-SNE Visualization (Every 5 epochs or last)
#     if epoch % 5 == 0 or epoch == FINETUNE_EPOCHS - 1:
#         board_embeddings, board_labels, proxy_embeddings, proxy_labels, all_probs = (
#             compute_tsne_embeddings(model, loss_fn_emb, val_loader, device)
#         )
#         fig = plot_tsne_embeddings(
#             board_embeddings,
#             board_labels,
#             proxy_embeddings,
#             all_probs,
#             title=f"Fine-Tune Epoch {epoch + 1}",
#             tag_names=TAG_NAMES,
#         )
#         writer.add_figure("Embeddings/TSNE", fig, global_step)
#         plt.close(fig)
#
# writer.close()
# print("Fine-tuning complete.")
