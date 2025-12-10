from collections import defaultdict
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, silhouette_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --- USER IMPORTS ---
from beschess.components.net.resnet import MultiTaskSEResEmbeddingNet
from beschess.components.net.vit import MultiTaskViT
from beschess.components.utils import clean_state_dict
from beschess.data.embedding import PuzzleDataset
from beschess.utils import tensor_to_board

np.random.seed(42)
torch.manual_seed(42)

# --- CONFIGURATION ---
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_INDICES_FILE = DATA_DIR / "test_indices.txt"

# Set this to True to test ViT, False for ResNet
USE_VIT = False

if USE_VIT:
    CHECKPOINT_PATH = (
        CHECKPOINT_DIR / "OptimizedModule_20251208_090924" / "checkpoint_epoch_25.pth"
    )
else:
    CHECKPOINT_PATH = (
        CHECKPOINT_DIR
        / "MultiTaskSEResEmbeddingNet_20251208_023154"
        / "best_checkpoint.pth"
    )

N_PUZZLES = 16000
N_QUIET = 16000
BATCH_SIZE = 64

TAG_NAMES = [
    "Quiet",
    "MatingNet",
    "SpecialMove",
    "Promotion",
    "DoubleAttack",
    "LinearAttack",
    "Punishment",
    "ForcingMove",
]


# --- METRIC 1: STRATIFIED RECALL ---
def compute_stratified_recall(embeddings, tags, k=5):
    """
    Calculates Recall@K specifically for each tag class.
    Assumes embeddings and tags ONLY contain Puzzles (no quiet boards).
    """
    print(f"\n--- Computing Stratified Recall@{k} ---")

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)

    class_scores = defaultdict(list)
    n_samples = len(embeddings)

    for i in range(n_samples):
        # Identify ground truth tags for this query
        query_tag_indices = np.where(tags[i] > 0)[0]

        if len(query_tag_indices) == 0:
            continue

        neighbor_indices = indices[i][1:]  # Skip self

        for t_idx in query_tag_indices:
            tag_name = TAG_NAMES[t_idx]

            # Count how many neighbors share this SPECIFIC tag
            hits = 0
            for n_idx in neighbor_indices:
                if tags[n_idx][t_idx] > 0:
                    hits += 1

            class_scores[tag_name].append(hits / k)

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


# --- METRIC 2: JACCARD SIMILARITY ---
def compute_jaccard_metrics(embeddings, tags, k=10):
    """
    Measures strict identity.
    J(A,B) = Intersection / Union.
    Punishes neighbors for having 'extra' tags.
    """
    print(f"\n--- Computing Jaccard Similarity@{k} ---")

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)

    jaccard_scores = []
    perfect_matches = 0
    total_queries = 0

    for i in range(len(embeddings)):
        query_set = set(np.where(tags[i] > 0)[0])
        if not query_set:
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


# --- METRIC 3: TAG COVERAGE ---
def compute_coverage_metrics(embeddings, tags, k=10):
    """
    Measures relevance.
    Coverage = Intersection / Size of Query.
    Does NOT punish neighbors for having 'extra' tags.
    """
    print(f"\n--- Computing Query Coverage@{k} ---")

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)

    coverage_scores = []

    for i in range(len(embeddings)):
        query_set = set(np.where(tags[i] > 0)[0])
        if not query_set:
            continue

        neighbor_indices = indices[i][1:]
        neighbor_coverages = []

        for n_idx in neighbor_indices:
            neighbor_set = set(np.where(tags[n_idx] > 0)[0])

            # Intersection / QUERY SIZE (Instead of Union)
            intersection = len(query_set & neighbor_set)
            score = intersection / len(query_set)
            neighbor_coverages.append(score)

        coverage_scores.append(np.mean(neighbor_coverages))

    avg_coverage = np.mean(coverage_scores) if coverage_scores else 0.0
    print(f"Global Average Coverage Score: {avg_coverage:.4f} (Higher is better)")


# --- METRIC 4: CLUSTER QUALITY (Structure) ---
def compute_cluster_quality(embeddings, tags):
    """
    Calculates Silhouette Score (Clustering) and Linear Separability (Semantics).
    """
    print("\n--- Computing Global Cluster Quality ---")

    # 1. Prepare Labels (Use primary/first tag for single-label metrics)
    primary_labels = np.argmax(tags, axis=1)

    # Filter valid samples
    valid_mask = np.sum(tags, axis=1) > 0
    clean_embeddings = embeddings[valid_mask]
    clean_labels = primary_labels[valid_mask]
    clean_tags = tags[valid_mask]

    if len(clean_embeddings) == 0:
        print("[CLUSTER] No valid tags found for cluster analysis.")
        return

    # 2. Silhouette Score (Subsampled for speed)
    # Measures how tight the clusters are.
    if len(clean_embeddings) > 5000:
        indices = np.random.choice(len(clean_embeddings), 5000, replace=False)
        sil_emb = clean_embeddings[indices]
        sil_lab = clean_labels[indices]
    else:
        sil_emb = clean_embeddings
        sil_lab = clean_labels

    try:
        # Check if we have at least 2 classes
        if len(np.unique(sil_lab)) > 1:
            sil_score = silhouette_score(sil_emb, sil_lab, metric="cosine")
            print(f"[CLUSTER] Silhouette Score: {sil_score:.4f} (Higher is better)")
            print(
                "          (-1 = Wrong Cluster, 0 = Overlapping, 1 = Perfect Separation)"
            )
        else:
            print("[CLUSTER] Silhouette Score: N/A (Only 1 class present)")
    except Exception as e:
        print(f"[CLUSTER] Silhouette Failed: {e}")

    # 3. Linear Probe (Logistic Regression)
    # Measures if classes are linearly separable.
    print(f"[CLUSTER] Training Linear Probe (Logistic Regression)...")

    X_train, X_test, y_train, y_test = train_test_split(
        clean_embeddings, clean_tags, test_size=0.3, random_state=42
    )

    scores = []
    # Train One-Vs-Rest for each tag
    for i in range(y_train.shape[1]):
        # Skip if not enough samples
        if np.sum(y_train[:, i]) < 10:
            continue

        clf = LogisticRegression(
            solver="liblinear", max_iter=1000, class_weight="balanced"
        )
        clf.fit(X_train, y_train[:, i])
        preds = clf.predict(X_test)

        score = f1_score(y_test[:, i], preds, zero_division=0)
        scores.append(score)

    avg_f1 = np.mean(scores) if scores else 0.0
    print(f"[CLUSTER] Linear Probe Mean F1: {avg_f1:.4f}")
    print("          (High score = Classes are distinct and separable)")


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

    p_indices = p_indices[:N_PUZZLES]
    q_indices = q_indices[:N_QUIET]
    print(f"Evaluating on {len(p_indices)} Puzzles and {len(q_indices)} Quiet Boards")

    puzzle_boards = np.load(DATA_DIR / "boards_packed.npy", mmap_mode="r")
    puzzle_tags = np.load(DATA_DIR / "tags_packed.npy", mmap_mode="r")
    quiet_boards = np.load(DATA_DIR / "quiet_boards_preeval.npy", mmap_mode="r")

    try:
        hard_negatives = np.load(DATA_DIR / "hard_negatives.npy", mmap_mode="r")
        # Balance Negatives logic...
        n_hard = len(hard_negatives)
        n_quiet = len(quiet_boards)
        if n_quiet > n_hard:
            indices = np.random.choice(n_quiet, n_hard, replace=False)
            quiet_subset = quiet_boards[indices]
            negatives_combined = np.concatenate([quiet_subset, hard_negatives], axis=0)
        else:
            negatives_combined = np.concatenate([quiet_boards, hard_negatives], axis=0)
    except FileNotFoundError:
        print("WARNING: hard_negatives.npy not found. Using Quiet Boards only.")
        negatives_combined = quiet_boards

    dataset = PuzzleDataset(negatives_combined, puzzle_boards, puzzle_tags)

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

            out = model(batch_tensor)
            if isinstance(out, tuple):
                embeddings_t, logits_t = out
            else:
                embeddings_t, logits_t = out, None

            embeddings_list.append(embeddings_t.cpu().numpy())
            if logits_t is not None:
                probs_list.append(torch.sigmoid(logits_t).cpu().numpy().flatten())

        all_tags = np.vstack(all_tags)
        embeddings = np.vstack(embeddings_list)
        if probs_list:
            probs = np.hstack(probs_list)
        else:
            probs = np.zeros(len(embeddings))

    # --- 4. EXECUTE METRICS (PUZZLES ONLY) ---
    print("\n--- Filtering: Removing Quiet Boards from Embedding Analysis ---")

    # Slice only the Puzzles
    n_puzzles_eval = len(p_indices)
    puzzle_embeddings = embeddings[:n_puzzles_eval]
    puzzle_tags = all_tags[:n_puzzles_eval]

    # A. Latent Space Collapse Check
    std_dev = np.std(puzzle_embeddings, axis=0).mean()
    print(f"[DIAGNOSTIC] Puzzle Embedding Std Dev: {std_dev:.4f}")

    # B. Auxiliary Task Accuracy & AUC
    if probs_list:
        # Create Ground Truth
        y_true = np.concatenate([np.ones(len(p_indices)), np.zeros(len(q_indices))])

        try:
            auc_score = roc_auc_score(y_true, probs)
            print(f"[AUX TASK] ROC AUC Score: {auc_score:.4f}")
        except ValueError:
            print("[AUX TASK] ROC AUC: N/A (Needs both classes)")

        p_probs = probs[:n_puzzles_eval]
        q_probs = probs[n_puzzles_eval:]
        print(
            f"[AUX TASK] Acc (0.5 thresh): Puzzle {(p_probs > 0.5).mean() * 100:.2f}% | Quiet {(q_probs < 0.5).mean() * 100:.2f}%"
        )

    # C. METRICS (On Puzzles Only)
    compute_stratified_recall(puzzle_embeddings, puzzle_tags, k=5)
    compute_jaccard_metrics(puzzle_embeddings, puzzle_tags, k=10)
    compute_coverage_metrics(puzzle_embeddings, puzzle_tags, k=10)

    # D. CLUSTER QUALITY (Structure Check)
    compute_cluster_quality(puzzle_embeddings, puzzle_tags)

    # E. VISUAL SANITY CHECK (With FENs)
    print("\n--- Visual Sanity Check (Puzzles Only + FENs) ---")

    nbrs = NearestNeighbors(n_neighbors=5, metric="cosine").fit(puzzle_embeddings)
    _, nn_indices = nbrs.kneighbors(puzzle_embeddings)

    sample_size = min(5, len(puzzle_embeddings))
    random_indices = np.random.choice(
        len(puzzle_embeddings), sample_size, replace=False
    )

    for i, embedding_idx in enumerate(random_indices):
        real_dataset_idx = p_indices[embedding_idx]
        query_tensor, _ = dataset[real_dataset_idx]
        query_tags = get_tag_string(puzzle_tags[embedding_idx])

        try:
            query_fen = tensor_to_board(query_tensor.numpy()).fen()
        except Exception as e:
            query_fen = f"Error generating FEN: {e}"

        print(f"\n{'=' * 60}")
        print(f"QUERY {i + 1}: {query_tags}")
        print(f"FEN: {query_fen}")
        print(f"{'-' * 60}")

        neighbor_embedding_idxs = nn_indices[embedding_idx][1:]

        for rank, n_emb_idx in enumerate(neighbor_embedding_idxs):
            n_real_idx = p_indices[n_emb_idx]
            n_tensor, _ = dataset[n_real_idx]
            n_tags = get_tag_string(puzzle_tags[n_emb_idx])

            try:
                n_fen = tensor_to_board(n_tensor.numpy()).fen()
            except:
                n_fen = "Error"

            print(f"  Rank {rank + 1}: {n_tags}")
            print(f"  FEN: {n_fen}")
            print(f"  {'-' * 20}")


if __name__ == "__main__":
    verify()
