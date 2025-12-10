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
# DATA_PATH = Path(__file__).resolve().parent.parent / "data"
# TOTAL_POSITIONS = 316_072_343
# TARGET_SAMPLES = 2_000_000
#
# MAX_DIFF = 100
# MIN_DEPTH = 20
#
# MIN_PIECES = 2
# MAX_PIECES = 32
# NUM_BUCKETS = (MAX_PIECES - MIN_PIECES) + 1
# SAMPLES_PER_BUCKET = TARGET_SAMPLES // NUM_BUCKETS
# TRUE_TOTAL_SAMPLES = SAMPLES_PER_BUCKET * NUM_BUCKETS
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
#     return sum(1 for c in board_part if c.isalpha())
#
#
# def quiet_boards_preeval(zstd_json_path, output_path):
#     boards_packed = np.zeros((TRUE_TOTAL_SAMPLES, 133), dtype=np.uint8)
#
#     current_samples = 0
#     seen_hashes = set()
#     bucket_counts = defaultdict(int)
#
#     print(
#         f"Mining {TRUE_TOTAL_SAMPLES} boards. Stratifying by Piece Count ({MIN_PIECES}-{MAX_PIECES})..."
#     )
#     print(f"Target per piece-count bucket: ~{SAMPLES_PER_BUCKET}")
#     print(f"Filter: Volatility < {MAX_DIFF}cp, Depth >= {MIN_DEPTH}")
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
#             evals_list = data.get("evals", [])
#             if not evals_list:
#                 continue
#
#             main_eval = evals_list[0]
#
#             if main_eval.get("depth", 0) < MIN_DEPTH:
#                 continue
#
#             pvs = main_eval.get("pvs", [])
#             if len(pvs) < 2:
#                 continue
#
#             s1 = calc_score(pvs[0])
#             s2 = calc_score(pvs[1])
#
#             if abs(s1 - s2) > MAX_DIFF:
#                 continue
#
#             fen = data["fen"]
#             piece_count = get_piece_count(fen)
#
#             if piece_count > MAX_PIECES:
#                 piece_count = MAX_PIECES
#             if piece_count < MIN_PIECES:
#                 piece_count = MIN_PIECES
#
#             if bucket_counts[piece_count] >= SAMPLES_PER_BUCKET:
#                 continue
#
#             fen_hash = hash(fen)
#             if fen_hash in seen_hashes:
#                 continue
#
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

import io
import sys
import ujson
import numpy as np
import chess
import multiprocessing
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

from tqdm import tqdm

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

from beschess.utils import board_to_packed

# --- CONFIGURATION ---
DATA_PATH = Path(__file__).resolve().parent.parent / "data"
TOTAL_POSITIONS = 316_072_343

# === HOW MANY TO SKIP ===
SKIP_POSITIONS = 11_000_000

TARGET_SAMPLES = 2_000_000
MAX_DIFF = 100
MIN_DEPTH = 20

MIN_PIECES = 2
MAX_PIECES = 32
NUM_BUCKETS = (MAX_PIECES - MIN_PIECES) + 1
SAMPLES_PER_BUCKET = TARGET_SAMPLES // NUM_BUCKETS
TRUE_TOTAL_SAMPLES = SAMPLES_PER_BUCKET * NUM_BUCKETS

BATCH_SIZE = 5_000
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)


# --- WORKER FUNCTIONS ---
def calc_score(pv):
    if "mate" in pv:
        return 30_000
    return int(pv.get("cp", 1000))


def get_piece_count(fen_str):
    board_part = fen_str.split(" ")[0]
    return sum(1 for c in board_part if c.isalpha())


def process_batch(lines):
    """
    Worker: Parses a batch of raw JSON lines.
    """
    candidates = []

    for line in lines:
        try:
            if not line:
                continue

            data = ujson.loads(line)

            evals_list = data.get("evals", [])
            if not evals_list:
                continue

            main_eval = evals_list[0]
            if main_eval.get("depth", 0) < MIN_DEPTH:
                continue

            pvs = main_eval.get("pvs", [])
            if len(pvs) < 2:
                continue

            s1 = calc_score(pvs[0])
            s2 = calc_score(pvs[1])

            if abs(s1 - s2) > MAX_DIFF:
                continue

            fen = data["fen"]
            piece_count = get_piece_count(fen)

            if piece_count > MAX_PIECES:
                piece_count = MAX_PIECES
            elif piece_count < MIN_PIECES:
                piece_count = MIN_PIECES

            candidates.append((fen, piece_count))

        except ValueError:
            continue
        except Exception:
            continue

    return candidates


def chunked_reader(reader, chunk_size):
    """Yields lists of lines."""
    batch = []
    for line in reader:
        batch.append(line)
        if len(batch) == chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch


# --- MAIN PROCESS ---


def quiet_boards_preeval(zstd_json_path, output_path):
    boards_packed = np.zeros((TRUE_TOTAL_SAMPLES, 133), dtype=np.uint8)

    current_samples = 0
    seen_hashes = set()
    bucket_counts = defaultdict(int)

    # Validate file path
    if not zstd_json_path.exists():
        print(f"Error: File not found at {zstd_json_path}")
        return

    print(f"Source: {zstd_json_path}")
    print(f"Total Lines: {TOTAL_POSITIONS}")
    print(f"Skipping First: {SKIP_POSITIONS}")
    print(f"Target: {TRUE_TOTAL_SAMPLES} samples")
    print(f"Workers: {MAX_WORKERS} | Batch Size: {BATCH_SIZE}")

    with zstd.open(zstd_json_path, "rb") as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")

        # --- PHASE 1: SKIPPING ---
        if SKIP_POSITIONS > 0:
            print("\nPhase 1: Fast-forwarding stream...")
            # We use a simple loop to consume the iterator
            # tqdm adds very slight overhead, but it's worth it for visibility
            for _ in tqdm(
                range(SKIP_POSITIONS), desc="Skipping", unit="lines", smoothing=0.1
            ):
                try:
                    next(text_stream)
                except StopIteration:
                    print("Error: EOF reached while skipping lines!")
                    return

        # --- PHASE 2: PROCESSING ---
        print("\nPhase 2: Mining quiet boards...")
        pbar_scan = tqdm(
            total=TOTAL_POSITIONS - SKIP_POSITIONS,
            desc="Scanning",
            unit="pos",
            smoothing=0.05,
        )
        pbar_saved = tqdm(
            total=TRUE_TOTAL_SAMPLES, desc="Saved", position=1, smoothing=0.05
        )

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}

            def handle_result(future):
                nonlocal current_samples
                b_size = futures.pop(future)
                pbar_scan.update(b_size)

                try:
                    candidates = future.result()
                except Exception:
                    return

                for fen, piece_count in candidates:
                    if current_samples >= TRUE_TOTAL_SAMPLES:
                        return

                    if bucket_counts[piece_count] >= SAMPLES_PER_BUCKET:
                        continue

                    fen_hash = hash(fen)
                    if fen_hash in seen_hashes:
                        continue

                    seen_hashes.add(fen_hash)
                    bucket_counts[piece_count] += 1

                    # Heavy lifting deferred to here
                    board = chess.Board(fen)
                    packed = board_to_packed(board)
                    boards_packed[current_samples] = packed

                    current_samples += 1
                    pbar_saved.update(1)

            # Read and submit loop
            for batch in chunked_reader(text_stream, BATCH_SIZE):
                if current_samples >= TRUE_TOTAL_SAMPLES:
                    break

                future = executor.submit(process_batch, batch)
                futures[future] = len(batch)

                # Backpressure control
                if len(futures) >= MAX_WORKERS * 2:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for f in done:
                        handle_result(f)

            # Clean up remaining futures
            while futures and current_samples < TRUE_TOTAL_SAMPLES:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for f in done:
                    handle_result(f)

    # Cleanup and Save
    if current_samples < TRUE_TOTAL_SAMPLES:
        boards_packed = boards_packed[:current_samples]
        print(f"\nWarning: EOF reached. Collected {current_samples} boards.")

    print(f"\nSaving to {output_path}...")
    np.save(output_path, boards_packed)

    pbar_scan.close()
    pbar_saved.close()

    print("\nFinal Distribution:")
    for k in sorted(bucket_counts.keys()):
        print(f"{k}: {bucket_counts[k]}")


if __name__ == "__main__":
    multiprocessing.freeze_support()

    raw_file = DATA_PATH / "raw" / "lichess_db_eval.jsonl.zst"
    out_file = DATA_PATH / "processed" / "quiet_boards_preeval_skipped.npy"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    quiet_boards_preeval(raw_file, out_file)
