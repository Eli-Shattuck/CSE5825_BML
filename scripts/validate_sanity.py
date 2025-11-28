import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

from beschess.components.net.resnet import MultiTaskSEResEmbeddingNet
from beschess.data.embedding import PuzzleDataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_INDICES_FILE = DATA_DIR / "test_indices.txt"

CHECKPOINT = (
    CHECKPOINT_DIR
    / "MultiTaskSEResEmbeddingNet_20251127_030440"
    / "best_checkpoint.pth"
)

N_PUZZLES = 8000
N_QUIET = 8000
BATCH_SIZE = 64

TAG_NAMES = [
    "Quiet",
    "LinearAttack",
    "DoubleAttack",
    "MatingNet",
    "Overload",
    "Displacement",
    "Sacrifice",
    "EndgameTactic",
    "PieceEndgame",
]


def compute_precision(embeddings, probs, tags, n_puzzles):
    """
    Calculates metrics assuming a production pipeline:
    1. Filter neighbors by Confidence > 0.8
    2. Check if the remaining neighbors match the Query Tag
    """
    print("\n--- RIGOROUS METRICS ---")
    from sklearn.neighbors import NearestNeighbors

    puzzle_embeddings = embeddings[:n_puzzles]
    puzzle_tags = tags[:n_puzzles]

    nbrs = NearestNeighbors(n_neighbors=10, metric="cosine").fit(embeddings)
    distances, indices = nbrs.kneighbors(puzzle_embeddings)

    precision_at_1 = 0.0
    precision_at_3 = 0.0
    valid_queries = 0

    confidence_threshold = 0.80

    for i in range(n_puzzles):
        query_tag_indices = np.where(puzzle_tags[i] > 0)[0]
        if len(query_tag_indices) == 0:
            continue

        candidate_indices = indices[i][1:]

        filtered_candidates = []
        for idx in candidate_indices:
            if probs[idx] > confidence_threshold:
                filtered_candidates.append(idx)

        if len(filtered_candidates) == 0:
            continue

        valid_queries += 1

        top_match = filtered_candidates[0]
        top_match_tags = np.where(tags[top_match] > 0)[0]

        if not set(query_tag_indices).isdisjoint(top_match_tags):
            precision_at_1 += 1

        hits_3 = 0
        denom = min(3, len(filtered_candidates))
        for k in range(denom):
            cand_idx = filtered_candidates[k]
            cand_tags = np.where(tags[cand_idx] > 0)[0]
            if not set(query_tag_indices).isdisjoint(cand_tags):
                hits_3 += 1

        precision_at_3 += hits_3 / denom

    p1 = precision_at_1 / valid_queries
    p3 = precision_at_3 / valid_queries

    print(
        f"Evaluated {valid_queries} queries (where at least 1 neighbor passed filter)"
    )
    print(f"Filtered Precision@1: {p1:.4f}")
    print(f"Filtered Precision@3: {p3:.4f}")

    return p1, p3


def get_tag_string(label_vector):
    """Converts a multi-hot vector (9,) into a readable list of strings."""
    active_indices = np.where(label_vector > 0)[0]
    tags = [TAG_NAMES[i] for i in active_indices]
    return tags


def verify():
    print(f"Device: {DEVICE}")

    print("Loading data...")
    p_indices = []
    q_indices = []

    with open(TEST_INDICES_FILE, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 2:
                continue

            type_flag, idx = parts[0], int(parts[1])
            if type_flag == "p":
                p_indices.append(idx)
            elif type_flag == "q":
                q_indices.append(idx)

    if N_PUZZLES and N_QUIET:
        import random

        random.seed(42)
        random.shuffle(p_indices)
        random.shuffle(q_indices)

        n_p = min(N_PUZZLES, len(p_indices))
        n_q = min(N_QUIET, len(q_indices))

        p_indices = p_indices[:n_p]
        q_indices = q_indices[:n_q]

        print(f"Evaluating on {n_p} Puzzles and {n_q} Quiet Boards")

    puzzle_boards = np.load(DATA_DIR / "boards_packed.npy", mmap_mode="r")
    puzzle_tags = np.load(DATA_DIR / "tags_packed.npy", mmap_mode="r")
    quiet_boards = np.load(DATA_DIR / "quiet_boards_preeval.npy", mmap_mode="r")

    dataset = PuzzleDataset(
        quiet_boards=quiet_boards,
        puzzle_boards=puzzle_boards,
        puzzle_labels=puzzle_tags,
    )

    print(f"Loading checkpoint: {CHECKPOINT.name}...")
    model = MultiTaskSEResEmbeddingNet(embedding_dim=128, num_blocks=10).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.eval()

    print("Running Inference...")

    with torch.no_grad():
        all_tags = []
        embeddings_list = []
        probs_list = []

        all_indices = p_indices + q_indices

        for start_idx in range(0, len(all_indices), BATCH_SIZE):
            batch_indices = all_indices[start_idx : start_idx + BATCH_SIZE]
            batch_boards = []

            for idx in batch_indices:
                board, label = dataset[idx]
                all_tags.append(label.numpy())
                batch_boards.append(board)

            batch_tensor = torch.stack(batch_boards).to(DEVICE)

            embeddings_t, logits_t = model(batch_tensor)
            probs_t = torch.sigmoid(logits_t)

            embeddings_list.append(embeddings_t.cpu().numpy())
            probs_list.append(probs_t.cpu().numpy().flatten())

        all_tags = np.vstack(all_tags)
        embeddings = np.vstack(embeddings_list)
        probs = np.hstack(probs_list)

    # --- METRIC 1: COLLAPSE ---
    std_dev = np.std(embeddings, axis=0).mean()
    print(f"\n[TEST SET] Embedding Std Dev: {std_dev:.4f}")
    if std_dev < 0.05:
        print(">>> FAIL: Model collapsed on test data.")
    else:
        print(">>> PASS: Variance is healthy.")

    # --- METRIC 2: BINARY ACCURACY ---
    # With all_indices = p_indices + q_indices:
    # First n_p indices are Puzzles.
    # Last n_q indices are Quiet.

    p_probs = probs[: len(p_indices)]
    q_probs = probs[len(p_indices) :]

    p_acc = (p_probs > 0.5).mean()
    q_acc = (q_probs < 0.5).mean()

    print(f"[TEST SET] Puzzle Detection Acc: {p_acc * 100:.2f}%")
    print(f"[TEST SET] Quiet Detection Acc:  {q_acc * 100:.2f}%")

    # --- METRIC 2b: FILTERED PRECISION@K ---
    compute_precision(embeddings, probs, all_tags, len(p_indices))

    # --- METRIC 3: SEMANTIC CONSISTENCY ---
    print("\nComputing Neighbors for Random Test Samples...")
    nbrs = NearestNeighbors(n_neighbors=6, metric="cosine").fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # Pick 5 random samples to display
    test_idx_selection = np.random.choice(len(all_indices), size=5, replace=False)

    for idx in test_idx_selection:
        q_tags = get_tag_string(all_tags[idx])
        q_conf = probs[idx]

        if idx < len(p_indices):
            q_type = "PUZZLE"
        else:
            q_type = "QUIET"

        print(f"\nQuery [{idx}] ({q_type}): {q_tags} (Conf: {q_conf * 100:.1f}%)")

        for rank, n_idx in enumerate(indices[idx][1:]):
            n_tags = get_tag_string(all_tags[n_idx])
            n_sim = 1.0 - distances[idx][rank + 1]
            n_conf = probs[n_idx]

            if n_idx < len(p_indices):
                n_type = "PUZZLE"
            else:
                n_type = "QUIET"

            print(
                f"  Rank {rank + 1} ({n_type}): {n_tags} | Sim: {n_sim:.3f} | Conf: {n_conf * 100:.1f}%"
            )


if __name__ == "__main__":
    verify()
