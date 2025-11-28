import numpy as np
import matplotlib.pyplot as plt
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
