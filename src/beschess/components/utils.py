from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from beschess.data.embedding import DirectLoader

from .loss import ProxyAnchor

TAG_NAMES = [
    "quiet",
    "bishopEndgame",
    "diagonalMate",
    "discoveredAttack",
    "fork",
    "knightEndgame",
    "knightMate",
    "orthogonalMate",
    "pawnEndgame",
    "pin",
    "queenEndgame",
    "queenMate",
    "queenRookEndgame",
    "rookEndgame",
    "skewer",
    "xRayAttack",
]


class CheckpointManager:
    def __init__(
        self,
        save_dir: Path | str,
        metric_key: str,
        higher_is_better: bool = True,
        save_interval: int = 10,
    ):
        self.save_dir = Path(save_dir)
        self.metric_key = metric_key
        self.higher_is_better = higher_is_better
        self.save_interval = save_interval

        self.best_metric = float("-inf") if higher_is_better else float("inf")

    def check(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        metrics: dict[str, float],
        epoch: int,
    ):
        current_metric = metrics.get(self.metric_key)
        if current_metric is None:
            raise ValueError(f"Metric '{self.metric_key}' not found in metrics.")

        is_better = (
            current_metric > self.best_metric
            if self.higher_is_better
            else current_metric < self.best_metric
        )

        if is_better:
            self.best_metric = current_metric
            self._save_checkpoint(
                model,
                loss_fn,
                optimizer,
                scheduler,
                metrics,
                "best_checkpoint.pth",
            )

        if epoch % self.save_interval == 0 and epoch != 0:
            file_name = f"checkpoint_epoch_{epoch}.pth"
            self._save_checkpoint(
                model,
                loss_fn,
                optimizer,
                scheduler,
                metrics,
                file_name,
            )

    def _save_checkpoint(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        metrics: dict[str, float],
        file_name: str,
    ):
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "loss_fn_state_dict": loss_fn.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
        }
        checkpoint_path = self.save_dir / file_name
        torch.save(checkpoint, checkpoint_path)


def compute_tsne_embeddings(
    model: nn.Module,
    loss_fn: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    n_samples: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model_is_train = model.training
    model.eval()

    all_embeddings = []
    all_labels = []
    all_probs = []

    count = 0
    with torch.no_grad():
        for boards, labels in dataloader:
            boards = boards.to(device)

            # Handle Multi-Task Output
            output = model(boards)
            if isinstance(output, tuple):
                embeddings, logits = output
                # Sigmoid to get probability [0, 1]
                probs = torch.sigmoid(logits).view(-1)
            else:
                # Fallback for single-head models (assume prob=1.0)
                embeddings = output
                probs = torch.ones(embeddings.size(0)).to(device)

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
            all_probs.append(probs.cpu())

            count += boards.size(0)
            if count >= n_samples:
                break

    model.train(model_is_train)

    all_embeddings = torch.cat(all_embeddings, dim=0)[:n_samples]
    all_embeddings = F.normalize(all_embeddings, p=2, dim=1).numpy()

    all_labels = torch.cat(all_labels, dim=0)[:n_samples].numpy()
    all_probs = torch.cat(all_probs, dim=0)[:n_samples].numpy()  # shape (N,)

    if hasattr(loss_fn, "proxies"):
        proxies = F.normalize(loss_fn.proxies.detach().cpu(), p=2, dim=1).numpy()
    else:
        proxies = np.zeros((1, all_embeddings.shape[1]))

    proxy_labels = np.arange(proxies.shape[0])

    # Run t-SNE
    X = np.vstack([all_embeddings, proxies])
    tsne_model = TSNE(
        n_components=2,
        init="pca",
        random_state=42,
    )
    X_embedded = tsne_model.fit_transform(X)

    embeddings_2d = X_embedded[:n_samples, :]
    proxies_2d = X_embedded[n_samples:, :]

    # Return 5 items now (added all_probs)
    return embeddings_2d, all_labels, proxies_2d, proxy_labels, all_probs


def plot_tsne_embeddings(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    proxies_2d: np.ndarray,
    probs: np.ndarray,
    title: str = "Latent Space",
    tag_names: list[str] = TAG_NAMES,
):
    """
    Visualizes t-SNE results.
    Points are transparent if the model thinks they are 'Quiet' (Low Prob).
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Setup Colormap for Tags (tab20)
    cmap = plt.get_cmap("tab20")
    # Normalize 0-19 to the colormap range
    norm = mcolors.Normalize(vmin=0, vmax=19)
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    # === PLOT A: Ghost Points (Multi-Label Detail) ===
    ax = axes[0]

    # 1. Plot Quiet Background (Index 0)
    quiet_mask = labels[:, 0] == 1

    if np.any(quiet_mask):
        # Quiet points: Use their calculated probability as alpha
        # Ideally this is near 0. We enforce a minimum visibility of 0.05
        # so they don't vanish entirely if you want to debug.
        q_probs = probs[quiet_mask]
        q_colors = np.zeros((len(q_probs), 4))  # RGBA
        q_colors[:, 0:3] = mcolors.to_rgb("lightgray")  # RGB
        q_colors[:, 3] = np.clip(q_probs, 0.05, 1.0)  # Alpha

        ax.scatter(
            embeddings_2d[quiet_mask, 0],
            embeddings_2d[quiet_mask, 1],
            c=q_colors,
            s=20,
            label="Quiet (Pred)",
            edgecolors="none",
        )

    # 2. Decompose Multi-Label Puzzles (Indices 1-15)
    # Get indices of active tags
    # labels shape: [N_samples, N_classes]
    tactical_rows, tactical_cols = np.nonzero(labels[:, 1:])

    # tactical_rows: indices of the original samples
    # tactical_cols: tag index (0..14). Shift by +1 to get global tag index (1..15)
    tactical_tags = tactical_cols + 1

    # 3. Jitter
    jitter_strength = 0.5
    jitter_x = np.random.uniform(
        -jitter_strength, jitter_strength, size=len(tactical_rows)
    )
    jitter_y = np.random.uniform(
        -jitter_strength, jitter_strength, size=len(tactical_rows)
    )

    # 4. Color & Alpha Construction
    # Get base RGB for each point based on its TAG
    point_rgbs = scalar_map.to_rgba(tactical_tags)[:, :3]

    # Get Alpha for each point based on its MODEL PROBABILITY
    # Note: We use tactical_rows to fetch the probability of the parent sample
    point_alphas = probs[tactical_rows]

    # Combine into RGBA
    point_rgba = np.column_stack((point_rgbs, point_alphas))

    ax.scatter(
        embeddings_2d[tactical_rows, 0] + jitter_x,
        embeddings_2d[tactical_rows, 1] + jitter_y,
        c=point_rgba,  # Matplotlib handles RGBA array natively
        s=10,
        edgecolors="none",
    )

    # 5. Plot Proxies
    for i, p in enumerate(proxies_2d):
        # Skip Proxy 0 (Quiet) if desired, or plot it to see where it is
        ax.scatter(p[0], p[1], c="black", marker="X", s=150, zorder=10)

        # Safe tag name indexing
        tag_name = tag_names[i] if i < len(tag_names) else str(i)

        ax.text(
            p[0],
            p[1],
            tag_name,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
        )

    ax.set_title(f"{title}: Ghost Points (Alpha = P(Puzzle))")
    ax.grid(True, alpha=0.3)

    # === PLOT B: Complexity Map (Cardinality) ===
    ax2 = axes[1]

    # Count active tags
    tag_counts = labels.sum(axis=1)
    tag_counts_capped = np.clip(tag_counts, 1, 3).astype(int)

    # Define Palette (RGB)
    # 0 (Quiet), 1, 2, 3+
    palette = np.array(
        [
            mcolors.to_rgb("#A0A0A0"),  # 0
            mcolors.to_rgb("cornflowerblue"),  # 1
            mcolors.to_rgb("gold"),  # 2
            mcolors.to_rgb("crimson"),  # 3
        ]
    )

    # Map counts to RGB
    c_rgb = palette[tag_counts_capped]

    # Combine with Probability Alpha
    c_rgba = np.column_stack((c_rgb, probs))

    ax2.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=c_rgba,
        s=15,
    )

    # Add Proxies to Plot B
    ax2.scatter(proxies_2d[:, 0], proxies_2d[:, 1], c="black", marker="X", s=100)

    # Manual Legend
    handles = [
        mpatches.Patch(color="cornflowerblue", label="1 Tag"),
        mpatches.Patch(color="gold", label="2 Tags"),
        mpatches.Patch(color="crimson", label="3+ Tags"),
        mpatches.Patch(color="lightgray", label="Quiet"),
    ]
    ax2.legend(handles=handles)
    ax2.set_title(f"{title}: Cardinality Map (Alpha = P(Puzzle))")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# def compute_tsne_embeddings(
#     model: nn.Module,
#     loss_fn: ProxyAnchor,
#     dataloader: DataLoader,
#     device: torch.device,
#     n_samples: int = 2000,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     model_is_train = model.training
#     model.eval()
#     all_embeddings = []
#     all_labels = []
#
#     count = 0
#     with torch.no_grad():
#         for boards, labels in dataloader:
#             boards = boards.to(device)
#             embeddings = model(boards)
#             if type(embeddings) is tuple:
#                 embeddings = embeddings[0]
#             all_embeddings.append(embeddings.cpu())
#             all_labels.append(labels)
#
#             count += boards.size(0)
#             if count >= n_samples:
#                 break
#
#     model.train(model_is_train)
#
#     all_embeddings = torch.cat(all_embeddings, dim=0)[:n_samples]
#     all_embeddings = F.normalize(all_embeddings, p=2, dim=1).numpy()
#     all_labels = torch.cat(all_labels, dim=0)[:n_samples].numpy()
#
#     proxies = F.normalize(loss_fn.proxies.detach().cpu(), p=2, dim=1).numpy()
#     proxy_labels = np.arange(proxies.shape[0])
#
#     X = np.vstack([all_embeddings, proxies])
#     tsne_model = TSNE(
#         n_components=2,
#         init="pca",
#         random_state=42,
#     )
#
#     X_embedded = tsne_model.fit_transform(X)
#     embeddings_2d = X_embedded[:n_samples, :]
#     proxies_2d = X_embedded[n_samples:, :]
#
#     return embeddings_2d, all_labels, proxies_2d, proxy_labels
#
# def plot_tsne_embeddings(
#     embeddings_2d: np.ndarray,
#     labels: np.ndarray,
#     proxies_2d: np.ndarray,
#     title: str = "Latent Space",
# ):
#     """
#     Visualizes t-SNE results using 'Jittered Ghost Points' to show multi-label overlaps.
#     """
#     fig, axes = plt.subplots(1, 2, figsize=(20, 10))
#     cmap = plt.get_cmap("tab20")
#
#     # === PLOT A: Ghost Points (Multi-Label Detail) ===
#     ax = axes[0]
#
#     # 1. Plot Quiet Background (Index 0)
#     quiet_mask = labels[:, 0] == 1
#     ax.scatter(
#         embeddings_2d[quiet_mask, 0],
#         embeddings_2d[quiet_mask, 1],
#         c="lightgray",
#         s=20,
#         alpha=0.3,
#         label="Quiet",
#         edgecolors="none",
#     )
#
#     # 2. Decompose Multi-Label Puzzles (Indices 1-15)
#     # np.nonzero returns (row_indices, col_indices)
#     # We only look at columns 1-15 (Tactical tags)
#     tactical_rows, tactical_cols = np.nonzero(labels[:, 1:])
#
#     # Adjust col index to match global tag index (0=Quiet, so Col 0 -> Tag 1)
#     tactical_tags = tactical_cols + 1
#
#     # 3. Apply Jitter
#     # We add random noise to separate the ghost points
#     jitter_strength = 0.5
#     jitter_x = np.random.uniform(
#         -jitter_strength, jitter_strength, size=len(tactical_rows)
#     )
#     jitter_y = np.random.uniform(
#         -jitter_strength, jitter_strength, size=len(tactical_rows)
#     )
#
#     # Plot the Ghost Cloud
#     ax.scatter(
#         embeddings_2d[tactical_rows, 0] + jitter_x,
#         embeddings_2d[tactical_rows, 1] + jitter_y,
#         c=tactical_tags,
#         cmap=cmap,
#         vmin=0,
#         vmax=19,  # Consistent color mapping with tab20
#         s=10,
#         alpha=0.6,
#         edgecolors="none",
#     )
#
#     # 4. Plot Proxies
#     for i, p in enumerate(proxies_2d):
#         ax.scatter(p[0], p[1], c="black", marker="X", s=150, zorder=10)
#         # Add text with box for readability
#         ax.text(
#             p[0],
#             p[1],
#             TAG_NAMES[i],
#             fontsize=9,
#             fontweight="bold",
#             ha="center",
#             va="center",
#             bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
#         )
#
#     ax.set_title(f"{title}: Ghost Points (Multi-Label)")
#     ax.grid(True, alpha=0.3)
#
#     # === PLOT B: Complexity Map (Cardinality) ===
#     ax2 = axes[1]
#
#     # Count active tags per puzzle
#     tag_counts = labels.sum(axis=1)
#
#     # Color scheme: 1=Blue, 2=Gold, 3+=Red
#     tag_counts_capped = np.clip(tag_counts, 1, 3).astype(int)
#     colors = np.array(
#         ["#A0A0A0", "cornflowerblue", "gold", "crimson"]
#     )  # 0(N/A), 1, 2, 3
#
#     # Map counts to colors
#     point_colors = colors[tag_counts_capped]
#
#     ax2.scatter(
#         embeddings_2d[:, 0], embeddings_2d[:, 1], c=point_colors, s=15, alpha=0.5
#     )
#
#     # Add Proxies to Plot B for reference
#     ax2.scatter(proxies_2d[:, 0], proxies_2d[:, 1], c="black", marker="X", s=100)
#
#     # Legend for B
#     handles = [
#         mpatches.Patch(color="cornflowerblue", label="1 Tag"),
#         mpatches.Patch(color="gold", label="2 Tags"),
#         mpatches.Patch(color="crimson", label="3+ Tags"),
#         mpatches.Patch(color="lightgray", label="Quiet (1 Tag)"),
#     ]
#     ax2.legend(handles=handles)
#     ax2.set_title(f"{title}: Cardinality Map")
#     ax2.grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     return fig


def warm_start_quiet_proxy(
    model: torch.nn.Module,
    loss_fn: ProxyAnchor,
    dataloader: torch.utils.data.DataLoader | DirectLoader,
    device: torch.device,
):
    model.eval()

    quiet_embeddings = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            quiet_mask = targets[:, 0] == 1.0

            if quiet_mask.sum() > 0:
                embedding = model(inputs[quiet_mask])
                if type(embedding) is tuple:
                    embedding = embedding[0]
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                quiet_embeddings.append(embedding)

            if len(quiet_embeddings) * dataloader.batch_size > 1000:
                break

    if not quiet_embeddings:
        print("Warning: No quiet samples found for warm start.")
        return

    all_quiet = torch.cat(quiet_embeddings, dim=0)
    centroid = all_quiet.mean(dim=0)
    centroid = torch.nn.functional.normalize(centroid, p=2, dim=0)

    loss_fn.proxies.data[0] = centroid


def lr_range_test(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    start_lr: float,
    end_lr: float,
    num_iters: int,
    beta: float = 0.05,
):
    for param_group in optimizer.param_groups:
        param_group["lr"] = start_lr

    def lr_lambda(iteration):
        return (end_lr / start_lr) ** (iteration / num_iters)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    iter_loader = iter(dataloader)

    lrs = []
    losses = []
    avg_loss = 0.0
    best_loss = float("inf")

    for iterations in tqdm(range(num_iters), desc="LR Range Test"):
        try:
            inputs, targets = next(iter_loader)
        except StopIteration:
            iter_loader = iter(dataloader)
            inputs, targets = next(iter_loader)

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        if type(outputs) is tuple:
            outputs = outputs[0]
        loss = loss_fn(outputs, targets)

        current_loss = loss.item()

        if iterations == 0:
            avg_loss = current_loss
        else:
            avg_loss = beta * current_loss + (1 - beta) * avg_loss

        if iterations > 0 and avg_loss > 4 * best_loss:
            print(f"Loss exploded at iteration {iterations}. Stopping.")
            break

        if avg_loss < best_loss:
            best_loss = avg_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)
        losses.append(avg_loss)

    return lrs, losses


def evaluate_proxy_cos(
    model: nn.Module,
    loss_fn: ProxyAnchor,
    dataloader: DataLoader,
    device: torch.device,
):
    model_is_train = model.training
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for boards, labels in tqdm(dataloader, desc="Evaluating Proxy COS"):
            boards = boards.to(device)
            embeddings = model(boards)
            if type(embeddings) is tuple:
                embeddings = embeddings[0]
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    std_dev = torch.std(torch.cat(all_embeddings, dim=0), dim=0).mean().item()
    print(f"Embedding Std Dev: {std_dev:.4f}")

    model.train(model_is_train)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    proxies = F.normalize(loss_fn.proxies.detach().cpu(), p=2, dim=1)
    all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
    similarity_matrix = torch.matmul(all_embeddings, proxies.T)

    return similarity_matrix, all_labels


def evaluate_knn_cos(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model_is_train = model.training
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for boards, labels in tqdm(dataloader, desc="Evaluating COS"):
            boards = boards.to(device)
            embeddings = model(boards)
            if type(embeddings) is tuple:
                embeddings = embeddings[0]
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    model.train(model_is_train)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
    similarity_matrix = torch.matmul(all_embeddings, all_embeddings.T)

    return similarity_matrix, all_labels


def compute_proxy_hitrate(
    top_indices: torch.Tensor, labels: torch.Tensor, k_values: list
) -> dict[int, float]:
    hitrates = {}

    hits = labels.gather(1, top_indices)

    cum_hits = hits.cumsum(dim=1)

    for k in k_values:
        has_hit = cum_hits[:, k - 1] > 0
        hitrates[k] = has_hit.float().mean().item()

    return hitrates


def compute_knn_hitrate(
    top_indices: torch.Tensor,
    labels: torch.Tensor,
    k_values: list,
) -> dict[int, float]:
    assert top_indices.size(0) == labels.size(0)
    assert top_indices.size(1) >= max(k_values) + 1

    n_samples = labels.size(0)
    recalls = {k: 0.0 for k in k_values}

    for k in k_values:
        # Exclude self-match at index 0
        pred_indices = top_indices[:, 1 : k + 1]
        pred_labels = labels[pred_indices]
        intersection = pred_labels * labels.unsqueeze(1)
        hits = (intersection.sum(dim=2) > 0).float()
        recalls[k] = (hits.sum(dim=1) > 0).float().sum().item()

    for k in k_values:
        recalls[k] /= n_samples

    return recalls


def compute_proxy_map(
    top_indices: torch.Tensor,
    labels: torch.Tensor,
    k_values: list,
) -> dict[int, float]:
    assert top_indices.size(0) == labels.size(0)
    assert top_indices.size(1) >= max(k_values)

    n_samples = labels.size(0)

    maps = {k: 0.0 for k in k_values}
    k_checkpoints = set(k_values)

    for i in range(n_samples):
        true_set = set(labels[i].nonzero(as_tuple=False).view(-1).tolist())
        num_true = len(true_set)

        if num_true == 0:
            continue

        hits = 0
        sum_precisions = 0.0

        pred_labels = top_indices[i].tolist()
        for j, pred_idx in enumerate(pred_labels):
            current_k = j + 1
            if pred_idx in true_set:
                hits += 1
                sum_precisions += hits / current_k

            if current_k in k_checkpoints:
                denom = min(num_true, current_k)
                if denom > 0:
                    maps[current_k] += sum_precisions / denom

    for k in k_values:
        maps[k] /= n_samples

    return maps


def compute_quiet_margin(
    similarity_matrix: torch.Tensor, labels: torch.Tensor
) -> tuple[float, float]:
    """
    Computes the 'Quiet Margin'

    This is defined as the average difference in similarity
    between the Quiet Proxy (index 0) and the most similar
    non-Quiet Proxy, across all Quiet samples.

    Larger margins indicate better separation.
    Higher accuracy indicates that the Quiet Proxy
    is the closest proxy more often.
    """
    quiet_mask = labels[:, 0] == 1.0

    if quiet_mask.sum() == 0:
        return 0.0, 0.0

    quiet_sims = similarity_matrix[quiet_mask]

    pos_scores = quiet_sims[:, 0]
    neg_scores = quiet_sims[:, 1:].max(dim=1).values

    margins = pos_scores - neg_scores

    avg_margin = margins.mean().item()

    accuracy = (margins > 0).float().mean().item()

    return avg_margin, accuracy
