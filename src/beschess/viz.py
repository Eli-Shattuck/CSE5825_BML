from typing import Iterable, Tuple

import chess
import chess.svg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances


def plot_distance_distributions(embeddings, labels, n_samples=5000):
    """Generates the Positive vs Negative Pair Histogram"""
    print("Generating Distance Histogram...")

    indices = np.random.choice(len(embeddings), n_samples, replace=False)
    X = embeddings[indices]
    # Mulihot label
    y = labels[indices]

    # Compute pairwise distances
    dists = cosine_distances(X)

    pos_dists = []
    neg_dists = []

    # Collect distances dealing with multi-hot labels
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            shared_tags = np.intersect1d(np.where(y[i] > 0)[0], np.where(y[j] > 0)[0])
            if len(shared_tags) > 0:
                pos_dists.append(dists[i, j])
            else:
                neg_dists.append(dists[i, j])

    plt.figure(figsize=(10, 6))
    sns.kdeplot(pos_dists, fill=True, color="g", label="Same Tag (Positive)", alpha=0.5)
    sns.kdeplot(neg_dists, fill=True, color="r", label="Diff Tag (Negative)", alpha=0.5)

    plt.title("Latent Space Separation: Positive vs Negative Pairs")
    plt.xlabel("Cosine Distance (Lower is closer)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()


def plot_jaccard_distance_distributions(embeddings, labels, n_samples=5000):
    """
    Generates a Latent Space Density plot split by Jaccard Similarity.

    Args:
        embeddings: (N, D) numpy array of latent vectors
        labels: (N, C) numpy array of multi-hot labels (0s and 1s)
    """
    print("Computing Jaccard-aware distances...")

    # 1. Random Sampling
    indices = np.random.choice(
        len(embeddings), min(n_samples, len(embeddings)), replace=False
    )
    X = embeddings[indices]
    y = labels[indices]

    # 2. Compute Cosine Distances (Vectorized)
    # Shape: (n_samples, n_samples)
    dist_matrix = cosine_distances(X)

    # 3. Compute Jaccard Similarity (Vectorized)
    # Intersection: Dot product of multi-hot vectors
    intersection = np.dot(y, y.T)

    # Union: |A| + |B| - |A n B|
    row_sums = y.sum(axis=1)
    # Broadcast sum to create matrix where cell [i,j] is len(y[i]) + len(y[j])
    cardinality_sum = row_sums[:, None] + row_sums[None, :]
    union = cardinality_sum - intersection

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard_matrix = intersection / union
        jaccard_matrix[union == 0] = 0.0  # Handle cases with 0 tags if any

    # 4. Flatten and Mask
    # We only care about the upper triangle (excluding diagonal self-comparisons)
    triu_indices = np.triu_indices_from(dist_matrix, k=1)

    flat_dists = dist_matrix[triu_indices]
    flat_jaccard = jaccard_matrix[triu_indices]

    # 5. Bucketing
    # Bucket A: Distinct (Jaccard == 0) - Theoretically "Negative"
    disjoint_mask = flat_jaccard == 0

    # Bucket B: Partial (0 < Jaccard < 1) - The "Hard Positives/Negatives"
    partial_mask = (flat_jaccard > 0) & (flat_jaccard < 1.0 - 1e-6)

    # Bucket C: Exact (Jaccard == 1) - The "True Positives"
    exact_mask = flat_jaccard >= (1.0 - 1e-6)

    disjoint_dists = flat_dists[disjoint_mask]
    partial_dists = flat_dists[partial_mask]
    exact_dists = flat_dists[exact_mask]

    # 6. Plotting
    plt.figure(figsize=(12, 7))

    # Plot Disjoint (Red)
    sns.kdeplot(
        disjoint_dists,
        fill=True,
        color="tab:red",
        label=f"Disjoint (J=0) [N={len(disjoint_dists)}]",
        alpha=0.3,
    )

    # Plot Partial (Orange) - THIS is where your "Blob" comes from
    if len(partial_dists) > 10:  # Only plot if we have enough samples
        sns.kdeplot(
            partial_dists,
            fill=True,
            color="tab:orange",
            label=f"Partial Overlap (0<J<1) [N={len(partial_dists)}]",
            alpha=0.4,
        )

    # Plot Exact (Green)
    if len(exact_dists) > 10:
        sns.kdeplot(
            exact_dists,
            fill=True,
            color="tab:green",
            label=f"Exact Semantics (J=1) [N={len(exact_dists)}]",
            alpha=0.5,
        )

    plt.title("Latent Space Separation by Jaccard Similarity")
    plt.xlabel("Cosine Distance (0.0 = Identical Embedding)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 2.0)  # Cosine distance range

    # Print stats to help debug
    print(f"Stats:\nExact Matches Mean Dist: {np.mean(exact_dists):.4f}")
    print(f"Partial Matches Mean Dist: {np.mean(partial_dists):.4f}")
    print(f"Disjoint Matches Mean Dist: {np.mean(disjoint_dists):.4f}")

    plt.show()


def plot_chessboard_overlay(
    board: chess.Board,
    overlay_vals: np.ndarray | None,
    query_idx: int | None = None,
    arrows: Iterable[chess.svg.Arrow | Tuple[chess.Square, chess.Square]] = [],
    size: int = 400,
    cmap: str = "viridis",
):
    if overlay_vals is not None:
        norm_att = (overlay_vals - np.min(overlay_vals)) / (
            np.max(overlay_vals) - np.min(overlay_vals) + 1e-8
        )
        att_colors = plt.get_cmap(cmap)(norm_att)
        hex_colors = [
            "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
            for r, g, b, _ in att_colors
        ]
        squares_dict = {square: hex_colors[i] for i, square in enumerate(chess.SQUARES)}
    else:
        squares_dict = {}

    focus_square = [chess.SQUARES[query_idx]] if query_idx is not None else []

    return chess.svg.board(
        board,
        fill=squares_dict,
        squares=focus_square,
        arrows=arrows,
        size=size,
    )


def plot_attention_overlay(
    board: chess.Board,
    attention_maps: np.ndarray,
    query_idx: int,
    head_idx: int,
    depth: int,
    size: int = 400,
    cmap: str = "viridis",
):
    offset_query_idx = query_idx - 1 if query_idx > 0 else None
    return plot_chessboard_overlay(
        board,
        attention_maps[depth, head_idx, query_idx][1:],
        offset_query_idx,
        size=size,
        cmap=cmap,
    )


def plot_attention_rollout_overlay(
    board: chess.Board,
    rollout_map: np.ndarray,
    query_idx: int,
    size: int = 400,
    cmap: str = "viridis",
):
    offset_query_idx = query_idx - 1 if query_idx > 0 else None
    return plot_chessboard_overlay(
        board,
        rollout_map[query_idx][1:],
        offset_query_idx,
        size=size,
        cmap=cmap,
    )


def plot_attention_move_overlay(
    board: chess.Board,
    attention_maps: np.ndarray,
    head_idx: int,
    depth: int,
    top_k: int = 3,
    size: int = 400,
    cmap: str = "viridis",
):
    att_map_no_cls = attention_maps[depth, head_idx, 1:, 1:]
    arrows = []
    norm_att = (att_map_no_cls - np.min(att_map_no_cls)) / (
        np.max(att_map_no_cls) - np.min(att_map_no_cls) + 1e-8
    )
    att_colors = plt.get_cmap(cmap)(norm_att)
    hex_colors = [
        [
            "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
            for r, g, b, _ in query_att_colors
        ]
        for query_att_colors in att_colors
    ]

    source_importance = att_map_no_cls.sum(axis=1)
    top_sources = np.argsort(-source_importance)[:top_k]

    for i in top_sources:
        max_attention = att_map_no_cls[i].max()
        if max_attention < 1e-2:
            continue

        threshold = 0.1 * max_attention
        for j in range(64):
            attention_stength = att_map_no_cls[i, j]

            if i == j or attention_stength < threshold:
                continue

            arrows.append(
                chess.svg.Arrow(
                    chess.SQUARES[i],
                    chess.SQUARES[j],
                    color=hex_colors[i][j],
                )
            )

    return plot_chessboard_overlay(
        board,
        None,
        None,
        arrows=arrows,
        size=size,
        cmap=cmap,
    )
