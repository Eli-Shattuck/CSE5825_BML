import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances
import chess
import chess.svg


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


def plot_chessboard_attention_overlay(
    board: chess.Board,
    attention_map: np.ndarray,
    query_idx: int,
    head_idx: int,
    depth: int,
    size: int = 400,
    cmap: str = "viridis",
):
    att = attention_map[depth, head_idx, query_idx][1:]
    norm_att = (att - np.min(att)) / (np.max(att) - np.min(att) + 1e-8)
    att_colors = plt.get_cmap(cmap)(norm_att)
    hex_colors = [
        "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b, _ in att_colors
    ]
    squares_dict = {square: hex_colors[i] for i, square in enumerate(chess.SQUARES)}
    focus_square = [chess.SQUARES[query_idx - 1]] if query_idx >= 0 else None
    print(query_idx, focus_square)

    return chess.svg.board(board, fill=squares_dict, squares=focus_square, size=size)
