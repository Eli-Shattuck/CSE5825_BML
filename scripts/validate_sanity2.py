from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

# --- USER IMPORTS (Assumed available in your environment) ---
from beschess.components.net.resnet import MultiTaskSEResEmbeddingNet
from beschess.components.net.vit import MultiTaskViT
from beschess.components.utils import clean_state_dict
from beschess.data.embedding import PuzzleDataset

# --- CONFIGURATION ---
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_INDICES_FILE = DATA_DIR / "test_indices.txt"

# Set this to True to test ViT, False for ResNet
USE_VIT = True

if USE_VIT:
    CHECKPOINT_PATH = (
        CHECKPOINT_DIR / "OptimizedModule_20251203_151336" / "best_checkpoint.pth"
    )
else:
    CHECKPOINT_PATH = (
        CHECKPOINT_DIR
        / "MultiTaskSEResEmbeddingNet_20251127_030440"
        / "best_checkpoint.pth"
    )

N_PUZZLES = 16000
N_QUIET = 16000
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


# --- METRIC 1: STRATIFIED RECALL (Architectural Bias) ---
def compute_stratified_recall(embeddings, tags, k=5):
    """
    Calculates Recall@K specifically for each tag class.
    This reveals if a model is 'blind' to specific tactics (e.g., Endgames).
    """
    print(f"\n--- Computing Stratified Recall@{k} ---")

    # Fit NN on all embeddings
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)

    class_scores = defaultdict(list)

    # We only care about Puzzles (tags > 0), not Quiet positions for this metric
    # Assuming the first N_PUZZLES indices are puzzles based on your loader logic
    n_puzzles = len(embeddings)  # Or limit this if you have a mix

    for i in range(n_puzzles):
        # Identify ground truth tags for this query
        query_tag_indices = np.where(tags[i] > 0)[0]

        # Skip if it's a "Quiet" position (assuming index 0 is Quiet in TAG_NAMES)
        # or if it has no tags
        if len(query_tag_indices) == 0 or (
            0 in query_tag_indices and len(query_tag_indices) == 1
        ):
            continue

        # Get neighbor indices (excluding self at index 0)
        neighbor_indices = indices[i][1:]

        for t_idx in query_tag_indices:
            tag_name = TAG_NAMES[t_idx]

            # Count how many neighbors share this SPECIFIC tag
            hits = 0
            for n_idx in neighbor_indices:
                if tags[n_idx][t_idx] > 0:
                    hits += 1

            # Score for this tag on this specific query (0.0 to 1.0)
            class_scores[tag_name].append(hits / k)

    # Convert to DataFrame for nice printing
    results = []
    for tag, scores in class_scores.items():
        results.append(
            {"Tag": tag, "Recall@K": np.mean(scores), "Samples": len(scores)}
        )

    df = pd.DataFrame(results)
    if not df.empty:
        print(
            df.sort_values("Recall@K", ascending=False).to_markdown(
                index=False, floatfmt=".4f"
            )
        )
    else:
        print("No puzzle tags found to evaluate.")


# --- METRIC 2: JACCARD SIMILARITY (Strictness) ---
def compute_jaccard_metrics(embeddings, tags, k=10):
    """
    Replaces 'Loose Match' with Jaccard Index.
    J(A,B) = |A intersect B| / |A union B|
    """
    print(f"\n--- Computing Jaccard Similarity@{k} ---")

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)

    jaccard_scores = []
    perfect_matches = 0
    total_queries = 0

    for i in range(len(embeddings)):
        query_set = set(np.where(tags[i] > 0)[0])

        # Skip Quiet queries for this specific tactical metric
        if 0 in query_set:
            continue

        total_queries += 1
        neighbor_indices = indices[i][1:]

        query_jaccards = []

        for n_idx in neighbor_indices:
            neighbor_set = set(np.where(tags[n_idx] > 0)[0])

            intersection = len(query_set & neighbor_set)
            union = len(query_set | neighbor_set)

            score = intersection / union if union > 0 else 0.0
            query_jaccards.append(score)

            if score == 1.0:
                perfect_matches += 1

        jaccard_scores.append(np.mean(query_jaccards))

    avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0.0
    perfect_rate = perfect_matches / (total_queries * k) if total_queries else 0.0

    print(f"Global Average Jaccard Score: {avg_jaccard:.4f} (1.0 is perfect)")
    print(
        f"Perfect Exact Matches Found:  {perfect_rate * 100:.2f}% of all recommendations"
    )
    return avg_jaccard


def get_tag_string(label_vector):
    active_indices = np.where(label_vector > 0)[0]
    return [TAG_NAMES[i] for i in active_indices]


def verify():
    print(f"Device: {DEVICE}")
    print(f"Mode: {'ViT' if USE_VIT else 'ResNet'}")

    # --- 1. DATA LOADING ---
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

    # Trim to size
    p_indices = p_indices[:N_PUZZLES]
    q_indices = q_indices[:N_QUIET]
    print(f"Evaluating on {len(p_indices)} Puzzles and {len(q_indices)} Quiet Boards")

    puzzle_boards = np.load(DATA_DIR / "boards_packed.npy", mmap_mode="r")
    puzzle_tags = np.load(DATA_DIR / "tags_packed.npy", mmap_mode="r")
    quiet_boards = np.load(DATA_DIR / "quiet_boards_preeval.npy", mmap_mode="r")

    dataset = PuzzleDataset(
        quiet_boards=quiet_boards,
        puzzle_boards=puzzle_boards,
        puzzle_labels=puzzle_tags,
    )

    # --- 2. MODEL LOADING ---
    print(f"Loading checkpoint: {CHECKPOINT_PATH.name}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )
    state_dict = clean_state_dict(state_dict)

    if USE_VIT:
        model = MultiTaskViT(
            in_channels=17, embed_dim=256, num_heads=8, depth=6, out_dim=128
        ).to(DEVICE)
    else:
        model = MultiTaskSEResEmbeddingNet(embedding_dim=128, num_blocks=10).to(DEVICE)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # --- 3. INFERENCE ---
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

            # Handle different return signatures if necessary
            out = model(batch_tensor)
            if isinstance(out, tuple):
                embeddings_t, logits_t = out
            else:
                embeddings_t, logits_t = out, None  # Handle models without aux head

            embeddings_list.append(embeddings_t.cpu().numpy())
            if logits_t is not None:
                probs_list.append(torch.sigmoid(logits_t).cpu().numpy().flatten())

        all_tags = np.vstack(all_tags)
        embeddings = np.vstack(embeddings_list)
        if probs_list:
            probs = np.hstack(probs_list)
        else:
            probs = np.zeros(len(embeddings))

    # --- 4. EXECUTE METRICS ---

    # A. Latent Space Collapse Check
    std_dev = np.std(embeddings, axis=0).mean()
    print(f"\n[DIAGNOSTIC] Embedding Std Dev: {std_dev:.4f} (Should be > 0.01)")

    # B. Auxiliary Task Accuracy (Only if probs exist)
    if probs_list:
        p_probs = probs[: len(p_indices)]
        q_probs = probs[len(p_indices) :]
        print(
            f"[AUX TASK] Puzzle Acc: {(p_probs > 0.5).mean() * 100:.2f}% | Quiet Acc: {(q_probs < 0.5).mean() * 100:.2f}%"
        )

    # C. NEW METRICS
    compute_stratified_recall(embeddings, all_tags, k=5)
    compute_jaccard_metrics(embeddings, all_tags, k=10)

    # D. Visual Sanity Check
    print("\n--- Visual Sanity Check (Random Samples) ---")
    nbrs = NearestNeighbors(n_neighbors=4, metric="cosine").fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)

    # Sample 3 Puzzles and 2 Quiet positions
    sample_idxs = np.concatenate(
        [
            np.random.choice(len(p_indices), 3, replace=False),
            np.random.choice(
                np.arange(len(p_indices), len(embeddings)), 2, replace=False
            ),
        ]
    )

    for idx in sample_idxs:
        q_tags = get_tag_string(all_tags[idx])
        q_type = "PUZZLE" if idx < len(p_indices) else "QUIET"
        print(f"\nQuery {idx} [{q_type}]: {q_tags}")

        for rank, n_idx in enumerate(indices[idx][1:]):
            n_tags = get_tag_string(all_tags[n_idx])
            n_type = "PUZZLE" if n_idx < len(p_indices) else "QUIET"
            print(f"  Rank {rank + 1} [{n_type}]: {n_tags}")


if __name__ == "__main__":
    verify()
