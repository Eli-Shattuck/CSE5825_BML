import random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm

# --- BESCHESS IMPORTS ---
from beschess.components.loss import ProxyAnchor
from beschess.components.net.vit import MultiTaskViT
from beschess.components.utils import (
    CheckpointManager,
    compute_proxy_hitrate,
    compute_proxy_map,
    compute_tsne_embeddings,
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
TAG_NAMES = [
    "MatingNet",
    "SpecialMove",
    "Promotion",
    "DoubleAttack",
    "LinearAttack",
    "Punishment",
    "ForcingMove",
]

# PATHS
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

# !!! POINT THIS TO YOUR STAGE 1 OUTPUT !!!
WARMUP_CHECKPOINT = (
    CHECKPOINT_DIR / "FineTune_7Class_20251206_222159" / "warmup_epoch_1.pt"
)

# HYPERPARAMETERS FOR A100
SEED = 42
EPOCHS = 10
# Increased Batch Size for A100 (Feed the tensor cores!)
# If you get OOM, reduce to 8192 or 4096
BATCH_SIZE = 16384
LAMBDA_BCE = 5.0
GRAD_CLIP = 1.0

LR_BACKBONE = 5e-5
LR_HEADS = 1e-4


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def evaluate_binary_accuracy(model, loader, device):
    """Calculates Binary Accuracy (Puzzle vs Non-Puzzle) on Validation Set"""
    model.eval()
    total_acc = 0.0
    num_batches = 0

    # We use Autocast in eval too for speed
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            is_puzzle_mask = targets[:, 0] == 0
            binary_targets = is_puzzle_mask.float().unsqueeze(1)

            _, puzzle_logits = model(inputs)
            preds = (torch.sigmoid(puzzle_logits) > 0.5).float()
            acc = (preds == binary_targets).float().mean()

            total_acc += acc.item()
            num_batches += 1

    return total_acc / num_batches if num_batches > 0 else 0.0


# ==========================================
# MAIN SCRIPT
# ==========================================
if __name__ == "__main__":
    # Setup
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # A100 Matmul Precision
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. LOAD DATA (RAM MODE)
    print("Loading data into RAM...")
    quiet_boards = np.load(
        DATA_DIR / "quiet_boards_preeval.npy"
    )  # Standard load (Fastest for training)
    puzzle_boards = np.load(DATA_DIR / "boards_packed.npy")
    puzzle_labels = np.load(DATA_DIR / "tags_packed.npy")

    try:
        hard_negatives = np.load(DATA_DIR / "hard_negatives.npy")
        print(f"Found {len(hard_negatives)} Hard Negatives.")

        # Balance Negatives
        n_hard = len(hard_negatives)
        n_quiet = len(quiet_boards)

        if n_quiet > n_hard:
            print(
                f"Downsampling Quiet Boards ({n_quiet}) to match Hard Negatives ({n_hard})..."
            )
            indices = np.random.choice(n_quiet, n_hard, replace=False)
            quiet_subset = quiet_boards[indices]
            negatives_combined = np.concatenate([quiet_subset, hard_negatives], axis=0)
        else:
            negatives_combined = np.concatenate([quiet_boards, hard_negatives], axis=0)
    except FileNotFoundError:
        print("WARNING: hard_negatives.npy not found. Using Quiet Boards only.")
        negatives_combined = quiet_boards

    dataset = PuzzleDataset(negatives_combined, puzzle_boards, puzzle_labels)
    splits = generate_split_indices(dataset)
    q_train, p_train = splits["train"]
    q_val, p_val = splits["val"]

    # Dataloaders - Increased Workers for high throughput
    train_loader = DataLoader(
        dataset,
        batch_sampler=BalancedBatchSampler(
            dataset, q_train, p_train, batch_size=BATCH_SIZE, steps_per_epoch=2000
        ),
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    # Validation loaders can be smaller/standard
    val_loader = DataLoader(
        dataset,
        batch_sampler=BalancedBatchSampler(
            dataset, q_val, p_val, batch_size=2048, steps_per_epoch=50
        ),
        num_workers=4,
    )
    val_puzzle_loader = DataLoader(
        Subset(dataset, p_val), batch_size=2048, num_workers=4
    )

    # 2. MODEL SETUP
    print("Initializing Model...")
    model = MultiTaskViT(
        in_channels=17, embed_dim=256, num_heads=8, depth=6, out_dim=128
    ).to(device)

    # Load Warmup Weights
    if WARMUP_CHECKPOINT.exists():
        print(f"Loading Warmup State from {WARMUP_CHECKPOINT}...")
        checkpoint = torch.load(WARMUP_CHECKPOINT, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    else:
        raise FileNotFoundError(f"Checkpoint not found at {WARMUP_CHECKPOINT}")

    # COMPILE (A100 OPTIMIZED)
    print("Compiling model (max-autotune)... this may take 2-3 minutes to start...")
    # max-autotune gives the highest throughput for GPUs
    model = torch.compile(model, mode="max-autotune")

    loss_fn_emb = ProxyAnchor(
        n_classes=len(TAG_NAMES), embedding_dim=128, margin=0.4, alpha=8
    ).to(device)
    loss_fn_binary = nn.BCEWithLogitsLoss().to(device)

    optimizer = optim.AdamW(
        [
            {"params": model.parameters(), "lr": LR_BACKBONE, "weight_decay": 1e-4},
            {"params": loss_fn_emb.parameters(), "lr": LR_HEADS, "weight_decay": 1e-4},
        ]
    )

    # Scaler for BF16 is optional but good practice to keep enabled=False or standard.
    # BFloat16 has huge range so scaling is rarely needed, but we use it for safety.
    # If this errors, set enabled=False.
    scaler = GradScaler(enabled=False)

    run_name = f"FineTune_Stage2_A100_{datetime.now().strftime('%Y%m%d_%H%M')}"
    (CHECKPOINT_DIR / run_name).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=LOG_DIR / run_name)
    checkpoint_manager = CheckpointManager(
        CHECKPOINT_DIR / run_name, metric_key="val_map@3", save_interval=1
    )

    # 3. TRAINING LOOP
    global_step = 0
    print("Starting Training...")

    for epoch in range(EPOCHS):
        model.train()
        loss_fn_emb.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)

        for inputs, targets in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            is_puzzle_mask = targets[:, 0] == 0
            binary_targets = is_puzzle_mask.float().unsqueeze(1)
            puzzle_targets = targets[is_puzzle_mask][:, 1:]

            optimizer.zero_grad()

            # --- A100 SPEED: BFloat16 AUTOCAST ---
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Forward (Full Batch)
                embeddings, puzzle_logits = model(inputs)

                # Losses
                loss_bce = loss_fn_binary(puzzle_logits, binary_targets)

                # Masking inside autocast
                if is_puzzle_mask.any():
                    # Slicing OUTPUT is safe for graph
                    loss_emd = loss_fn_emb(embeddings[is_puzzle_mask], puzzle_targets)
                else:
                    loss_emd = torch.tensor(0.0, device=device)

                loss = loss_emd + (LAMBDA_BCE * loss_bce)

            # Backward
            # Note: With enabled=False scaler, these just pass through
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            # Sparse Logging (Every 50 steps) to avoid CPU bottlenecks
            if global_step % 50 == 0:
                # Need to detach and float() to bring to CPU for logging
                # Keep calc inside no_grad to save memory
                with torch.no_grad():
                    preds = (torch.sigmoid(puzzle_logits) > 0.5).float()
                    acc = (preds == binary_targets).float().mean()

                    l_item = loss.item()
                    a_item = acc.item()

                writer.add_scalar("Train/Loss", l_item, global_step)
                writer.add_scalar("Train/Binary_Acc", a_item, global_step)
                writer.add_scalar("Train/Loss_Proxy", loss_emd.item(), global_step)

                pbar.set_postfix({"Loss": f"{l_item:.2f}", "Acc": f"{a_item:.2f}"})

            global_step += 1

        # --- EVALUATION ---
        print(f"Evaluating Epoch {epoch + 1}...")

        # 1. Binary Accuracy
        val_bin_acc = evaluate_binary_accuracy(model, val_loader, device)

        # 2. Metric Performance
        model.eval()
        # Evaluate Proxy Cos handles its own no_grad/logic
        # We wrap the internal forward pass of evaluate_proxy_cos with autocast if possible
        # but for simplicity we assume it runs fast enough in default precision or
        # you can modify evaluate_proxy_cos to accept an autocast context.
        # Here we just run it as is (Evaluations are rarely the bottleneck).
        similarity_matrix, val_labels_emb = evaluate_proxy_cos(
            model, loss_fn_emb, val_puzzle_loader, device
        )

        similarity_matrix = similarity_matrix.cpu()
        val_labels_emb = val_labels_emb.cpu()[:, 1:]

        _, top_indices = torch.topk(similarity_matrix, k=3, dim=1)
        k_list = [1, 3]
        val_map = compute_proxy_map(top_indices, val_labels_emb, k_list)
        hitrate = compute_proxy_hitrate(top_indices, val_labels_emb, k_list)

        print(f"Epoch {epoch + 1} Results:")
        print(f"  > MAP@3: {val_map[3]:.4f}")
        print(f"  > HR@1:  {hitrate[1]:.4f}")
        print(f"  > Bin Acc: {val_bin_acc:.4f}")

        writer.add_scalar("Val/MAP@3", val_map[3], global_step)
        writer.add_scalar("Val/Binary_Acc", val_bin_acc, global_step)

        checkpoint_manager.check(
            model, loss_fn_emb, optimizer, None, {"val_map@3": val_map[3]}, epoch
        )

        # 3. T-SNE (Only last epoch to save time)
        if epoch == EPOCHS - 1:
            print("Generating T-SNE...")
            b_emb, b_lab, p_emb, p_lab, probs = compute_tsne_embeddings(
                model, loss_fn_emb, val_loader, device
            )
            fig = plot_tsne_embeddings(
                b_emb,
                b_lab,
                p_emb,
                probs,
                title=f"Stage 2 Epoch {epoch + 1}",
                tag_names=TAG_NAMES,
            )
            writer.add_figure("Embeddings/TSNE", fig, global_step)
            plt.close(fig)

    writer.close()
    print("Stage 2 Complete.")
