import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances


def plot_distance_distributions(embeddings, labels, n_samples=5000):
    """Generates the Positive vs Negative Pair Histogram"""
    print("Generating Distance Histogram...")

    # Subsample for speed
    indices = np.random.choice(len(embeddings), n_samples, replace=False)
    X = embeddings[indices]
    y = np.argmax(labels[indices], axis=1)  # Use primary tag for simplification

    # Compute pairwise distances
    dists = cosine_distances(X)

    pos_dists = []
    neg_dists = []

    # Collect distances
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            d = dists[i, j]
            if y[i] == y[j]:
                pos_dists.append(d)
            elif y[i] != y[j]:  # Different tag
                neg_dists.append(d)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(pos_dists, fill=True, color="g", label="Same Tag (Positive)", alpha=0.5)
    sns.kdeplot(neg_dists, fill=True, color="r", label="Diff Tag (Negative)", alpha=0.5)

    plt.title("Latent Space Separation: Positive vs Negative Pairs")
    plt.xlabel("Cosine Distance (Lower is closer)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()
