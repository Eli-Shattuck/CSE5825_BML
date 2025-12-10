from pathlib import Path

import chess
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from beschess.utils import board_to_packed

DATA_PATH = Path(__file__).resolve().parent.parent / "data"

# 1. VISUAL CORE CLASSES
CORE_CLASSES = [
    "MatingNet",  # Rigid King-restriction geometry
    "SpecialMove",  # Relies on Channels 13-17 (Castling/EP)
    "Promotion",  # Relies on Pawn Rank (Row 0/7)
    "DoubleAttack",  # Divergent Geometry (V-shape)
    "LinearAttack",  # Alignment Geometry (I-shape)
    "Punishment",  # Static Weakness (Undefended/Trapped)
    "ForcingMove",  # Dynamic/Active moves (includes Sacrifices)
]

# 2. TAG MAPPING
TAG_MAPPING = {
    # --- Priority 1: MATING PATTERNS ---
    "mate": "MatingNet",
    "mateIn1": "MatingNet",
    "mateIn2": "MatingNet",
    "mateIn3": "MatingNet",
    "mateIn4": "MatingNet",
    "mateIn5": "MatingNet",
    "anastasiaMate": "MatingNet",
    "arabianMate": "MatingNet",
    "backRankMate": "MatingNet",
    "bodensMate": "MatingNet",
    "doubleBishopMate": "MatingNet",
    "dovetailMate": "MatingNet",
    "hookMate": "MatingNet",
    "smotheredMate": "MatingNet",
    # --- Priority 2: SPECIAL RULES  ---
    "enPassant": "SpecialMove",
    "castling": "SpecialMove",
    # --- Priority 3: PROMOTION ---
    "promotion": "Promotion",
    "underPromotion": "Promotion",
    "advancedPawn": "Promotion",
    # --- Priority 4/5: GEOMETRY ---
    "fork": "DoubleAttack",
    "discoveredAttack": "DoubleAttack",
    "doubleCheck": "DoubleAttack",
    "pin": "LinearAttack",
    "skewer": "LinearAttack",
    "xRayAttack": "LinearAttack",
    # --- Priority 6: STATIC WEAKNESS ---
    "hangingPiece": "Punishment",
    "trappedPiece": "Punishment",
    "exposedKing": "Punishment",
    # --- Priority 7: DYNAMIC/COMPLEX --
    "attraction": "ForcingMove",
    "deflection": "ForcingMove",
    "interference": "ForcingMove",
    "sacrifice": "ForcingMove",
    "clearance": "ForcingMove",
    "intermezzo": "ForcingMove",
    "capturingDefender": "ForcingMove",
}


def clean_and_map_tags(tag_str):
    if not isinstance(tag_str, str):
        return []

    raw_tags = tag_str.split()
    mapped_tags = set()

    for t in raw_tags:
        if t in TAG_MAPPING:
            mapped_tags.add(TAG_MAPPING[t])

    return list(mapped_tags)


def main():
    print(f"Loading {DATA_PATH / 'lichess_db_puzzle.csv'}...")
    df = pd.read_csv(
        DATA_PATH / "lichess_db_puzzle.csv", usecols=["FEN", "Themes", "Moves"]
    )

    print(f"Mapping Tags to {len(CORE_CLASSES)} Core Clusters...")
    df["clean_tags"] = df["Themes"].apply(clean_and_map_tags)

    initial_len = len(df)
    df = df[df["clean_tags"].map(len) > 0].reset_index(drop=True)
    print(
        f"Dropped {initial_len - len(df)} puzzles that did not fit the Core Taxonomy."
    )

    mlb = MultiLabelBinarizer(classes=CORE_CLASSES)
    tags_matrix = mlb.fit_transform(df["clean_tags"]).astype(np.uint8)

    np.save(DATA_PATH / "processed" / "tag_classes.npy", mlb.classes_)
    print(f"Classes: {mlb.classes_}")

    print("Bit-Packing Boards (Int8)...")
    num_samples = len(df)
    boards_packed = np.zeros((num_samples, 133), dtype=np.int8)

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

    print("Saving...")
    np.save(DATA_PATH / "processed" / "boards_packed.npy", boards_packed)
    np.save(DATA_PATH / "processed" / "tags_packed.npy", tags_matrix)
    print("Complete.")


if __name__ == "__main__":
    main()
