# from pathlib import Path
#
# import chess
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MultiLabelBinarizer
# from tqdm import tqdm
#
# from beschess.utils import board_to_packed
#
# DATA_PATH = Path(__file__).resolve().parent.parent / "data"
#
# # 1. VISUAL CORE CLASSES
# CORE_CLASSES = [
#     "MatingNet",  # Rigid King-restriction geometry
#     "SpecialMove",  # Relies on Channels 13-17 (Castling/EP)
#     "Promotion",  # Relies on Pawn Rank (Row 0/7)
#     "DoubleAttack",  # Divergent Geometry (V-shape)
#     "LinearAttack",  # Alignment Geometry (I-shape)
#     "Punishment",  # Static Weakness (Undefended/Trapped)
#     "ForcingMove",  # Dynamic/Active moves (includes Sacrifices)
# ]
#
# # 2. TAG MAPPING
# TAG_MAPPING = {
#     # --- Priority 1: MATING PATTERNS ---
#     "mate": "MatingNet",
#     "mateIn1": "MatingNet",
#     "mateIn2": "MatingNet",
#     "mateIn3": "MatingNet",
#     "mateIn4": "MatingNet",
#     "mateIn5": "MatingNet",
#     "anastasiaMate": "MatingNet",
#     "arabianMate": "MatingNet",
#     "backRankMate": "MatingNet",
#     "bodensMate": "MatingNet",
#     "doubleBishopMate": "MatingNet",
#     "dovetailMate": "MatingNet",
#     "hookMate": "MatingNet",
#     "smotheredMate": "MatingNet",
#     # --- Priority 2: SPECIAL RULES  ---
#     "enPassant": "SpecialMove",
#     "castling": "SpecialMove",
#     # --- Priority 3: PROMOTION ---
#     "promotion": "Promotion",
#     "underPromotion": "Promotion",
#     "advancedPawn": "Promotion",
#     # --- Priority 4/5: GEOMETRY ---
#     "fork": "DoubleAttack",
#     "discoveredAttack": "DoubleAttack",
#     "doubleCheck": "DoubleAttack",
#     "pin": "LinearAttack",
#     "skewer": "LinearAttack",
#     "xRayAttack": "LinearAttack",
#     # --- Priority 6: STATIC WEAKNESS ---
#     "hangingPiece": "Punishment",
#     "trappedPiece": "Punishment",
#     "exposedKing": "Punishment",
#     # --- Priority 7: DYNAMIC/COMPLEX --
#     "attraction": "ForcingMove",
#     "deflection": "ForcingMove",
#     "interference": "ForcingMove",
#     "sacrifice": "ForcingMove",
#     "clearance": "ForcingMove",
#     "intermezzo": "ForcingMove",
#     "capturingDefender": "ForcingMove",
# }
#
#
# def clean_and_map_tags(tag_str):
#     if not isinstance(tag_str, str):
#         return []
#
#     raw_tags = tag_str.split()
#     mapped_tags = set()
#
#     for t in raw_tags:
#         if t in TAG_MAPPING:
#             mapped_tags.add(TAG_MAPPING[t])
#
#     return list(mapped_tags)
#
#
# def main():
#     print(f"Loading {DATA_PATH / 'lichess_db_puzzle.csv'}...")
#     df = pd.read_csv(
#         DATA_PATH / "lichess_db_puzzle.csv", usecols=["FEN", "Themes", "Moves"]
#     )
#
#     print(f"Mapping Tags to {len(CORE_CLASSES)} Core Clusters...")
#     df["clean_tags"] = df["Themes"].apply(clean_and_map_tags)
#
#     initial_len = len(df)
#     df = df[df["clean_tags"].map(len) > 0].reset_index(drop=True)
#     print(
#         f"Dropped {initial_len - len(df)} puzzles that did not fit the Core Taxonomy."
#     )
#
#     mlb = MultiLabelBinarizer(classes=CORE_CLASSES)
#     tags_matrix = mlb.fit_transform(df["clean_tags"]).astype(np.uint8)
#
#     np.save(DATA_PATH / "processed" / "tag_classes.npy", mlb.classes_)
#     print(f"Classes: {mlb.classes_}")
#
#     print("Bit-Packing Boards (Int8)...")
#     num_samples = len(df)
#     boards_packed = np.zeros((num_samples, 133), dtype=np.uint8)
#
#     for i, (fen, move_str) in tqdm(
#         enumerate(zip(df["FEN"], df["Moves"])), total=num_samples
#     ):
#         board = chess.Board(fen)
#         try:
#             first_move_uci = move_str.split(" ")[0]
#             board.push_uci(first_move_uci)
#         except:
#             continue
#         boards_packed[i] = board_to_packed(board)
#
#     print("Saving...")
#     np.save(DATA_PATH / "processed" / "boards_packed.npy", boards_packed)
#     np.save(DATA_PATH / "processed" / "tags_packed.npy", tags_matrix)
#     print("Complete.")
#
#
# if __name__ == "__main__":
#     main()

import sys
import numpy as np
import pandas as pd
import chess
import multiprocessing
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Ensure beschess is in path if running as script
from beschess.utils import board_to_packed

# --- CONFIGURATION ---
DATA_PATH = Path(__file__).resolve().parent.parent / "data"

# Batch size for multiprocessing (adjust based on RAM)
BATCH_SIZE = 50_000
# Leave 1 core for the main process
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)


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


# --- WORKER FUNCTION ---
def process_board_batch(data_chunk):
    """
    Process a tuple of (fens, moves).
    Returns a numpy array of packed boards.
    """
    fens, moves = data_chunk
    count = len(fens)
    # Pre-allocate batch array
    batch_packed = np.zeros((count, 133), dtype=np.uint8)

    for i, (fen, move_str) in enumerate(zip(fens, moves)):
        try:
            board = chess.Board(fen)
            first_move_uci = move_str.split(" ")[0]
            board.push_uci(first_move_uci)
            batch_packed[i] = board_to_packed(board)
        except:
            # If invalid, leave as zeros (matching original script behavior)
            pass

    return batch_packed


def main():
    csv_path = DATA_PATH / "lichess_db_puzzle.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, usecols=["FEN", "Themes", "Moves"])

    print(f"Mapping Tags to {len(CORE_CLASSES)} Core Clusters...")
    df["clean_tags"] = df["Themes"].apply(clean_and_map_tags)

    initial_len = len(df)
    # Filter rows with no valid tags
    df = df[df["clean_tags"].map(len) > 0].reset_index(drop=True)
    print(
        f"Dropped {initial_len - len(df)} puzzles that did not fit the Core Taxonomy."
    )

    # 1. Process Tags (Fast enough on main thread)
    print("Binarizing Tags...")
    mlb = MultiLabelBinarizer(classes=CORE_CLASSES)
    tags_matrix = mlb.fit_transform(df["clean_tags"]).astype(np.uint8)

    # Save Tag Classes Immediately
    (DATA_PATH / "processed").mkdir(parents=True, exist_ok=True)
    np.save(DATA_PATH / "processed" / "tag_classes.npy", mlb.classes_)
    print(f"Classes: {mlb.classes_}")

    # 2. Prepare Data for Multiprocessing
    print(f"Bit-Packing Boards with {MAX_WORKERS} workers...")

    fens = df["FEN"].tolist()
    moves = df["Moves"].tolist()
    num_samples = len(df)

    # Create chunks generator
    chunk_indices = range(0, num_samples, BATCH_SIZE)
    chunks = [
        (fens[i : i + BATCH_SIZE], moves[i : i + BATCH_SIZE]) for i in chunk_indices
    ]

    boards_packed_list = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map returns results in order, so alignment is preserved
        results = list(
            tqdm(
                executor.map(process_board_batch, chunks),
                total=len(chunks),
                unit="batch",
            )
        )

        # Aggregate results
        # vstack is fast for a list of arrays
        boards_packed = np.vstack(results)

    print(f"Final shape: {boards_packed.shape}")

    print("Saving to disk...")
    np.save(DATA_PATH / "processed" / "boards_packed.npy", boards_packed)
    np.save(DATA_PATH / "processed" / "tags_packed.npy", tags_matrix)
    print("Complete.")


if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    main()
