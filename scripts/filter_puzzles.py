from pathlib import Path

import chess
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from beschess.utils import board_to_packed

DATA_PATH = Path(__file__).resolve().parent.parent / "data"

df = pd.read_csv(DATA_PATH / "lichess_db_puzzle.csv")

KEPT_THEMES = {
    "pin",
    "skewer",
    "fork",
    "discoveredAttack",
    "xRayAttack",
    "pawnEndgame",
    "rookEndgame",
    "queenEndgame",
    "bishopEndgame",
    "knightEndgame",
    "queenRookEndgame",
    "diagonalMate",
    "orthogonalMate",
    "knightMate",
    "queenMate",
}


MERGE_THEMES = {
    # Merge diagonal mating thems into "diagonalMate"
    "bodenMate": "diagonalMate",
    "doubleBishopMate": "diagonalMate",
    "balestraMate": "diagonalMate",
    # Merge plus shaped mating themes into "orthogonalMate"
    "blindSwineMate": "orthogonalMate",
    "anastasiaMate": "orthogonalMate",
    "arabianMate": "orthogonalMate",
    "backRankMate": "orthogonalMate",
    # Merge knight mating themes into "knightMate"
    "smotheredMate": "knightMate",
    "hookMate": "knightMate",
    "vukovicMate": "knightMate",
    # Merge queen mating themes into "queenMate"
    "dovetailMate": "queenMate",
    "triangleMate": "queenMate",
    "killBoxMate": "queenMate",
}

# The final valid vocabulary is KEPT_THEMES + the targets of MERGE_THEMES
VALID_TAGS = KEPT_THEMES.union(set(MERGE_THEMES.values()))


def clean_and_map_tags(tag_str):
    if not isinstance(tag_str, str):
        return []

    tags = [MERGE_THEMES.get(t, t) for t in tag_str.split()]
    return [t for t in tags if t in VALID_TAGS]


def main():
    print(f"Loading {DATA_PATH / 'lichess_db_puzzle.csv'}...")
    df = pd.read_csv(
        DATA_PATH / "lichess_db_puzzle.csv", usecols=["FEN", "Themes", "Moves"]
    )

    print("Processing Tags...")
    # Apply the cleaning function
    df["clean_tags"] = df["Themes"].apply(clean_and_map_tags)

    # Drop rows that have NO valid tags after filtering (Visual Noise)
    initial_len = len(df)
    df = df[df["clean_tags"].map(len) > 0].reset_index(drop=True)
    print(f"Dropped {initial_len - len(df)} puzzles with no visual tags.")

    # Multi-Hot Encode
    mlb = MultiLabelBinarizer()
    tags_matrix = mlb.fit_transform(df["clean_tags"]).astype(np.int8)

    # Save the class names so you know Index 0 = 'bishopEndgame'
    np.save(DATA_PATH / "processed" / "tag_classes.npy", mlb.classes_)
    print(f"Encoded {len(mlb.classes_)} unique classes: {mlb.classes_}")

    print("Bit-Packing Boards (Int8)...")
    # Pre-allocate (N, 64) array
    num_samples = len(df)
    boards_packed = np.zeros((num_samples, 69), dtype=np.int8)

    # Loop with progress bar
    for i, (fen, move_str) in tqdm(
        enumerate(zip(df["FEN"], df["Moves"])), total=num_samples
    ):
        board = chess.Board(fen)
        try:
            first_move_uci = move_str.split(" ")[0]
            board.push_uci(first_move_uci)
        except:
            continue

        boards_packed[i] = board_to_packed(board)

    print("Saving to disk...")
    # Save uncompressed for mmap speed
    np.save(DATA_PATH / "processed" / "boards_packed.npy", boards_packed)
    np.save(DATA_PATH / "processed" / "tags_packed.npy", tags_matrix)
    print("Done. Ready for training.")


if __name__ == "__main__":
    main()
