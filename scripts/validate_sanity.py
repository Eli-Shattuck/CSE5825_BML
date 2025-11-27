import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from beschess.components.net.resnet import SEResEmbeddingNet, MultiTaskSEResEmbeddingNet
from beschess.components.loss import ProxyAnchor
from beschess.utils import packed_to_tensor, packed_to_board
from pathlib import Path

# Config
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CHECKPOINT = CHECKPOINT_DIR / "BCE_ResNet_20251125_232349" / "checkpoint_10.pt"
# CHECKPOINT = (
#     CHECKPOINT_DIR / "SEResEmbeddingNet_20251126_000940" / "best_checkpoint.pth"
# )
# CHECKPOINT = (
#     CHECKPOINT_DIR / "SEResEmbeddingNet_20251126_000940" / "checkpoint_epoch_10.pth"
# )
# CHECKPOINT = (
#     CHECKPOINT_DIR / "SEResEmbeddingNet_20251126_161405" / "best_checkpoint.pth"
# )
CHECKPOINT = (
    CHECKPOINT_DIR
    / "MultiTaskSEResEmbeddingNet_20251126_173236"
    / "best_checkpoint.pth"
)
N_SAMPLES = 5000  # Load a small chunk to test


def verify():
    # 1. Load Data
    print("Loading data subset...")
    quiet_boards_packed = np.load(DATA_DIR / "quiet_boards_preeval.npy", mmap_mode="r")[
        : N_SAMPLES // 2
    ]
    boards_packed = np.load(DATA_DIR / "boards_packed.npy", mmap_mode="r")[:N_SAMPLES]
    labels_packed = np.load(DATA_DIR / "tags_packed.npy", mmap_mode="r")[:N_SAMPLES]

    # 2. Load Model
    print("Loading model...")
    # model = SEResEmbeddingNet(embedding_dim=128, num_blocks=10, reduction=8).to(DEVICE)
    model = MultiTaskSEResEmbeddingNet(embedding_dim=128, num_blocks=10).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )
    model.eval()

    # 3. Generate Embeddings
    print("Generating embeddings...")
    embeddings = []
    # Process in one batch for speed (5000 fits in VRAM)
    puzzle_tensors = np.array([packed_to_tensor(b) for b in boards_packed])
    quiet_tensors = np.array([packed_to_tensor(b) for b in quiet_boards_packed])
    tensors = np.concatenate((puzzle_tensors, quiet_tensors), axis=0)
    inputs = torch.from_numpy(tensors).float().to(DEVICE)

    with torch.no_grad():
        out = model(inputs)
        if isinstance(out, tuple):
            out = out[0]
        embeddings = out.cpu().numpy()

    # 4. Find Nearest Neighbors
    print("Computing Neighbors...")
    nbrs = NearestNeighbors(n_neighbors=4, metric="cosine").fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # 5. Visual Inspection
    # We define tag names (Update this list to match your actual 15 tags!)
    tag_names = [
        "quiet",
        "LinearAttack",
        "DoubleAttack",
        "MatingNet",
        "Overload",
        "Displacement",
        "Sacrifice",
        "EndgameTactic",
        "PieceEndgame",
    ]

    print("\n--- RESULTS ---")
    # Pick 5 random puzzles to inspect
    test_indices = np.random.choice(len(embeddings), 10, replace=False)

    for idx in test_indices:
        # Decode ground truth tags
        # query_tags = [tag_names[i] for i, x in enumerate(labels_packed[idx]) if x > 0]
        query_tags = []
        if idx >= len(boards_packed):
            query_tags.append("quiet")
        else:
            for i, x in enumerate(labels_packed[idx]):
                if x > 0:
                    query_tags.append(tag_names[i])

        print(f"\nQuery Puzzle {idx}: {query_tags}")

        # Show Top 3 Neighbors (skipping index 0 which is itself)
        neighbor_idxs = indices[idx][1:]

        for rank, n_idx in enumerate(neighbor_idxs):
            # n_tags = [tag_names[i] for i, x in enumerate(labels_packed[n_idx]) if x > 0]
            n_tags = []
            if n_idx >= len(boards_packed):
                n_tags.append("quiet")
            else:
                for i, x in enumerate(labels_packed[n_idx]):
                    if x > 0:
                        n_tags.append(tag_names[i])
            dist = distances[idx][rank + 1]
            print(f"  Neighbor {rank + 1} (Dist {dist:.3f}): {n_tags}")


def verify_loss_fn():
    print("Loading model...")
    loss_fn = ProxyAnchor(
        n_classes=16,
        embedding_dim=128,
        margin=0.1,
        alpha=32,
    ).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    loss_fn.load_state_dict(
        checkpoint["loss_fn_state_dict"]
        if "loss_fn_state_dict" in checkpoint
        else checkpoint
    )
    loss_fn.eval()
    proxies = loss_fn.proxies  # (16, 128)

    # Normalize (Proxies live on the sphere)
    proxies = torch.nn.functional.normalize(proxies, p=2, dim=1)

    # Compute Similarity Matrix
    sim_matrix = torch.matmul(proxies, proxies.T)

    # Print specific pair: DiscoveredAttack vs MateIn1 (Assuming indices)
    # You can print the whole matrix or the mean off-diagonal value
    print(f"Mean Proxy Similarity: {sim_matrix.mean().item():.4f}")
    print(
        f"Max Similarity (Worst overlap): {sim_matrix.fill_diagonal_(0).max().item():.4f}"
    )


if __name__ == "__main__":
    verify()
    # verify_loss_fn()
