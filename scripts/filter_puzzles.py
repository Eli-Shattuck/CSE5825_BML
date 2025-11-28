from pathlib import Path

import chess
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from beschess.utils import board_to_packed

DATA_PATH = Path(__file__).resolve().parent.parent / "data"

CORE_CLASSES = [
    "LinearAttack",
    "DoubleAttack",
    "MatingNet",
    "Overload",
    "Displacement",
    "Sacrifice",
    "EndgameTactic",
    "PieceEndgame",
]

# Merge classes into core clusters
TAG_MAPPING = {
    # Single Piece Attacks
    "pin": "LinearAttack",
    "skewer": "LinearAttack",
    "xRayAttack": "LinearAttack",
    # Piece Attacks on Multiple Targets
    "fork": "DoubleAttack",
    "discoveredAttack": "DoubleAttack",
    "doubleCheck": "DoubleAttack",
    # Matting Patterns
    "mate": "MatingNet",
    "mateIn1": "MatingNet",
    "mateIn2": "MatingNet",
    "mateIn3": "MatingNet",
    "mateIn4": "MatingNet",
    "mateIn5": "MatingNet",
    "anastasiaMate": "MatingNet",
    "arabianMate": "MatingNet",
    "backRankMate": "MatingNet",
    "balestraMate": "MatingNet",
    "blindSwineMate": "MatingNet",
    "bodenMate": "MatingNet",
    "cornerMate": "MatingNet",
    "doubleBishopMate": "MatingNet",
    "dovetailMate": "MatingNet",
    "hookMate": "MatingNet",
    "killBoxMate": "MatingNet",
    "smotheredMate": "MatingNet",
    "triangleMate": "MatingNet",
    "vukovicMate": "MatingNet",
    # Forcing Opponent to Defend Multiple Threats
    "attraction": "Overload",
    "trappedPiece": "Overload",
    "hangingPiece": "Overload",
    "exposedKing": "Overload",
    # Positional/Tactical Gain by Moving Pieces but not Capturing
    "deflection": "Displacement",
    "interference": "Displacement",
    # Giving up Material for Positional/Tactical Gain
    "sacrifice": "Sacrifice",
    "clearance": "Sacrifice",
    "intermezzo": "Sacrifice",
    "capturingDefender": "Sacrifice",
    # Tactics that often appear in endgames
    "promotion": "EndgameTactic",
    "underPromotion": "EndgameTactic",
    "zugzwang": "EndgameTactic",
    "advancedPawn": "EndgameTactic",
    "enPassant": "EndgameTactic",
    # Endgame Types
    "pawnEndgame": "PieceEndgame",
    "rookEndgame": "PieceEndgame",
    "bishopEndgame": "PieceEndgame",
    "knightEndgame": "PieceEndgame",
    "queenEndgame": "PieceEndgame",
    "queenRookEndgame": "PieceEndgame",
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

    print("Mapping Tags to 8 Core Clusters...")
    df["clean_tags"] = df["Themes"].apply(clean_and_map_tags)

    # Filter empty rows
    initial_len = len(df)
    df = df[df["clean_tags"].map(len) > 0].reset_index(drop=True)
    print(
        f"Dropped {initial_len - len(df)} puzzles that did not fit the Core Taxonomy."
    )

    # Encode
    mlb = MultiLabelBinarizer(classes=CORE_CLASSES)
    tags_matrix = mlb.fit_transform(df["clean_tags"]).astype(np.int8)

    # Save Class Names
    np.save(DATA_PATH / "processed" / "tag_classes.npy", mlb.classes_)
    print(f"Classes: {mlb.classes_}")

    print("Bit-Packing Boards (Int8)...")
    num_samples = len(df)
    # 69 bytes = 64 squares + 5 meta
    boards_packed = np.zeros((num_samples, 69), dtype=np.int8)

    for i, (fen, move_str) in tqdm(
        enumerate(zip(df["FEN"], df["Moves"])), total=num_samples
    ):
        board = chess.Board(fen)
        try:
            # Apply first move to get the position the user actually sees/solves
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
