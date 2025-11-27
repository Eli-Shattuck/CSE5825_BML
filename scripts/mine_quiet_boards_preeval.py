# import io
# import numpy as np
# import sys
# from pathlib import Path
# from collections import defaultdict
#
# import chess
# import ujson
# from tqdm import tqdm
#
# if sys.version_info >= (3, 14):
#     from compression import zstd
# else:
#     from backports import zstd
#
# from beschess.utils import board_to_packed
#
# # --- Configuration ---
# DATA_PATH = Path(__file__).resolve().parent.parent / "data"
# TOTAL_POSITIONS = 316_072_343
# TARGET_SAMPLES = 1_000_000
# MAX_EVAL_CP = 50
# MAX_DIFF_CP = 50
#
# # --- NEW: Material Stratification Settings ---
# # We bucket by number of pieces on board (2 kings to 32 total units)
# # This acts as a proxy for game phase.
# MIN_PIECES = 2
# MAX_PIECES = 32
# NUM_BUCKETS = (MAX_PIECES - MIN_PIECES) + 1
# SAMPLES_PER_BUCKET = TARGET_SAMPLES // NUM_BUCKETS
# TRUE_TOTAL_SAMPLES = SAMPLES_PER_BUCKET * NUM_BUCKETS
#
# # Heuristic: We want slightly more middlegame positions than 2-piece endgames
# # But strict uniform distribution is fine for now.
#
#
# def calc_score(pv):
#     if "mate" in pv:
#         return 30_000
#     return int(pv.get("cp", 1000))
#
#
# def get_piece_count(fen_str):
#     """
#     Counts pieces in the FEN string (first field).
#     Proxies game phase: High count = Opening, Low count = Endgame.
#     """
#     board_part = fen_str.split(" ")[0]
#     # fast count of letters (pieces)
#     return sum(1 for c in board_part if c.isalpha())
#
#
# def quiet_boards_preeval(zstd_json_path, output_path):
#     boards_packed = np.zeros((TRUE_TOTAL_SAMPLES, 69), dtype=np.int8)
#
#     current_samples = 0
#     seen_hashes = set()
#     bucket_counts = defaultdict(int)
#
#     print(
#         f"Mining {TRUE_TOTAL_SAMPLES} boards. Stratifying by Piece Count ({MIN_PIECES}-{MAX_PIECES})..."
#     )
#     print(f"Target per piece-count bucket: ~{SAMPLES_PER_BUCKET}")
#
#     pbar_scan = tqdm(total=TOTAL_POSITIONS, desc="Scanning positions", unit="pos")
#     pbar_saved = tqdm(total=TRUE_TOTAL_SAMPLES, desc="Saved Quiet Boards", position=1)
#
#     with zstd.open(zstd_json_path, "rb") as reader:
#         text_stream = io.TextIOWrapper(reader, encoding="utf-8")
#
#         for line in text_stream:
#             pbar_scan.update(1)
#
#             try:
#                 data = ujson.loads(line)
#             except ValueError:
#                 continue
#
#             # --- A. Check Evaluation (Quietness) ---
#             evals_list = data.get("evals", [])
#             if not evals_list:
#                 continue
#
#             main_eval = evals_list[0]
#             pvs = main_eval.get("pvs", [])
#             if len(pvs) < 2:
#                 continue
#
#             s1 = calc_score(pvs[0])
#             s2 = calc_score(pvs[1])
#
#             if abs(s1) > MAX_EVAL_CP:
#                 continue
#             if abs(s1 - s2) > MAX_DIFF_CP:
#                 continue
#
#             # --- B. Stratification (Piece Count) ---
#             # We use piece count as the bucket index directly
#             fen = data["fen"]
#             piece_count = get_piece_count(fen)
#
#             # Clamp just in case of malformed FENs
#             if piece_count > MAX_PIECES:
#                 piece_count = MAX_PIECES
#             if piece_count < MIN_PIECES:
#                 piece_count = MIN_PIECES
#
#             # Check if we have enough samples for this specific "complexity level"
#             if bucket_counts[piece_count] >= SAMPLES_PER_BUCKET:
#                 continue
#
#             # --- C. Deduplication ---
#             fen_hash = hash(fen)
#             if fen_hash in seen_hashes:
#                 continue
#
#             # --- Save ---
#             seen_hashes.add(fen_hash)
#             bucket_counts[piece_count] += 1
#
#             board = chess.Board(fen)
#             packed = board_to_packed(board)
#             boards_packed[current_samples] = packed
#             current_samples += 1
#             pbar_saved.update(1)
#
#             if current_samples >= TRUE_TOTAL_SAMPLES:
#                 break
#
#     if current_samples < TRUE_TOTAL_SAMPLES:
#         boards_packed = boards_packed[:current_samples]
#         print(f"Warning: EOF reached. Collected {current_samples} boards.")
#
#     print(f"Saving to {output_path}...")
#     np.save(output_path, boards_packed)
#
#     pbar_scan.close()
#     pbar_saved.close()
#
#     # Debug: Show distribution
#     print("\nFinal Distribution (Pieces on board -> Count):")
#     sorted_keys = sorted(bucket_counts.keys())
#     for k in sorted_keys:
#         print(f"{k}: {bucket_counts[k]}")
#
#
# if __name__ == "__main__":
#     raw_file = DATA_PATH / "raw" / "lichess_db_eval.jsonl.zst"
#     out_file = DATA_PATH / "processed" / "quiet_boards_preeval.npy"
#     quiet_boards_preeval(raw_file, out_file)

# 3: 23809
# 4: 64516
# 5: 64516
# 6: 64516
# 7: 64516
# 8: 64516
# 9: 64516
# 10: 64516
# 11: 64516
# 12: 64516
# 13: 64516
# 14: 64516
# 15: 64516
# 16: 64516
# 17: 64516
# 18: 64516
# 19: 64516
# 20: 64516
# 21: 64516
# 22: 64516
# 23: 64516
# 24: 64516
# 25: 64516
# 26: 64516
# 27: 64516
# 28: 64516
# 29: 64516
# 30: 64516
# 31: 64516
# 32: 64516

import io
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict

import chess
import ujson
from tqdm import tqdm

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

from beschess.utils import board_to_packed

# --- Configuration ---
DATA_PATH = Path(__file__).resolve().parent.parent / "data"
TOTAL_POSITIONS = 316_072_343
TARGET_SAMPLES = 2_000_000

# --- UPDATED FILTER SETTINGS ---
# We no longer cap the evaluation (MAX_EVAL_CP is removed).
# We want winning positions that are "boring" (Technical wins).
MAX_VOLATILITY = 100  # Max centipawn difference between Top Move and 2nd Move
MIN_DEPTH = 20  # Minimum engine depth to trust the evaluation

# --- Material Stratification Settings ---
MIN_PIECES = 2
MAX_PIECES = 32
NUM_BUCKETS = (MAX_PIECES - MIN_PIECES) + 1
SAMPLES_PER_BUCKET = TARGET_SAMPLES // NUM_BUCKETS
TRUE_TOTAL_SAMPLES = SAMPLES_PER_BUCKET * NUM_BUCKETS


def calc_score(pv):
    if "mate" in pv:
        return 30_000
    return int(pv.get("cp", 1000))


def get_piece_count(fen_str):
    """
    Counts pieces in the FEN string (first field).
    Proxies game phase: High count = Opening, Low count = Endgame.
    """
    board_part = fen_str.split(" ")[0]
    return sum(1 for c in board_part if c.isalpha())


def quiet_boards_preeval(zstd_json_path, output_path):
    boards_packed = np.zeros((TRUE_TOTAL_SAMPLES, 69), dtype=np.int8)

    current_samples = 0
    seen_hashes = set()
    bucket_counts = defaultdict(int)

    print(
        f"Mining {TRUE_TOTAL_SAMPLES} boards. Stratifying by Piece Count ({MIN_PIECES}-{MAX_PIECES})..."
    )
    print(f"Target per piece-count bucket: ~{SAMPLES_PER_BUCKET}")
    print(f"Filter: Volatility < {MAX_VOLATILITY}cp, Depth >= {MIN_DEPTH}")

    pbar_scan = tqdm(total=TOTAL_POSITIONS, desc="Scanning positions", unit="pos")
    pbar_saved = tqdm(total=TRUE_TOTAL_SAMPLES, desc="Saved Quiet Boards", position=1)

    with zstd.open(zstd_json_path, "rb") as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")

        for line in text_stream:
            pbar_scan.update(1)

            try:
                data = ujson.loads(line)
            except ValueError:
                continue

            # --- A. Check Evaluation (Quietness) ---
            evals_list = data.get("evals", [])
            if not evals_list:
                continue

            main_eval = evals_list[0]

            # 1. Depth Filter: Ignore shallow evaluations (unreliable for quietness)
            if main_eval.get("depth", 0) < MIN_DEPTH:
                continue

            pvs = main_eval.get("pvs", [])
            # We need at least 2 lines to compare volatility
            if len(pvs) < 2:
                continue

            s1 = calc_score(pvs[0])  # Best move score
            s2 = calc_score(pvs[1])  # 2nd Best move score

            # 2. Exclude Active Mates (Inherently Tactical)
            # If s1 is > 15000, someone is being mated.
            if abs(s1) > 15000:
                continue

            # 3. Volatility Check (The Critical Filter)
            # If BestMove is significantly better than 2ndBest, it is a "Forcing Move".
            # We want positions where the gap is small (Quiet).
            if abs(s1 - s2) > MAX_VOLATILITY:
                continue

            # --- B. Stratification (Piece Count) ---
            fen = data["fen"]
            piece_count = get_piece_count(fen)

            if piece_count > MAX_PIECES:
                piece_count = MAX_PIECES
            if piece_count < MIN_PIECES:
                piece_count = MIN_PIECES

            if bucket_counts[piece_count] >= SAMPLES_PER_BUCKET:
                continue

            # --- C. Deduplication ---
            fen_hash = hash(fen)
            if fen_hash in seen_hashes:
                continue

            # --- Save ---
            seen_hashes.add(fen_hash)
            bucket_counts[piece_count] += 1

            board = chess.Board(fen)
            packed = board_to_packed(board)
            boards_packed[current_samples] = packed
            current_samples += 1
            pbar_saved.update(1)

            if current_samples >= TRUE_TOTAL_SAMPLES:
                break

    if current_samples < TRUE_TOTAL_SAMPLES:
        boards_packed = boards_packed[:current_samples]
        print(f"Warning: EOF reached. Collected {current_samples} boards.")

    print(f"Saving to {output_path}...")
    np.save(output_path, boards_packed)

    pbar_scan.close()
    pbar_saved.close()

    print("\nFinal Distribution (Pieces on board -> Count):")
    sorted_keys = sorted(bucket_counts.keys())
    for k in sorted_keys:
        print(f"{k}: {bucket_counts[k]}")


if __name__ == "__main__":
    raw_file = DATA_PATH / "raw" / "lichess_db_eval.jsonl.zst"
    out_file = DATA_PATH / "processed" / "quiet_boards_preeval.npy"
    quiet_boards_preeval(raw_file, out_file)
