import io
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
import chess
import ujson
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

# Assuming this exists in your project
from beschess.utils import board_to_packed

# --- CONFIGURATION ---
DATA_PATH = Path(__file__).resolve().parent.parent / "data"
TOTAL_POSITIONS = 316_072_343
TARGET_SAMPLES = 2_000_000

MAX_CP_SCORE = 85
MIN_DEPTH = 20
MIN_PIECES = 4
MAX_PIECES = 32

NUM_BUCKETS = (MAX_PIECES - MIN_PIECES) + 1
SAMPLES_PER_BUCKET = TARGET_SAMPLES // NUM_BUCKETS
TRUE_TOTAL_SAMPLES = SAMPLES_PER_BUCKET * NUM_BUCKETS

# Worker batch size
BATCH_SIZE = 500
MAX_WORKERS = None


def calc_score(pv):
    if "mate" in pv:
        return 30_000 if pv["mate"] > 0 else -30_000
    return int(pv.get("cp", 1000))


def get_piece_count(fen_str):
    board_part = fen_str.split(" ")[0]
    return sum(1 for c in board_part if c.isalpha())


def process_batch(fens):
    """
    Worker process
    """
    results = []
    for fen in fens:
        # Wrap in try/except to prevent one bad FEN from crashing the worker
        try:
            board = chess.Board(fen)
            has_flashy = False
            for move in board.legal_moves:
                if board.gives_check(move) or board.is_capture(move) or move.promotion:
                    has_flashy = True
                    break

            if has_flashy:
                packed = board_to_packed(board)
                results.append(packed)
            else:
                results.append(None)
        except Exception:
            results.append(None)
    return results


def mine_hard_negatives_multiprocess(zstd_json_path, output_path):
    boards_packed = np.zeros((TRUE_TOTAL_SAMPLES, 133), dtype=np.uint8)

    current_samples = 0
    seen_hashes = set()
    bucket_counts = defaultdict(int)

    batch_fens = []
    batch_indices = []

    print(f"Mining {TRUE_TOTAL_SAMPLES} Hard Negatives using Multiprocessing...")

    pbar_scan = tqdm(total=TOTAL_POSITIONS, desc="Scanning (Main)", unit="pos")
    pbar_saved = tqdm(total=TRUE_TOTAL_SAMPLES, desc="Saved (Workers)", position=1)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        is_finished = False  # Flag to signal we are done

        with zstd.open(zstd_json_path, "rb") as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")

            for line in text_stream:
                if is_finished:
                    break  # Stop reading file if done

                pbar_scan.update(1)

                try:
                    data = ujson.loads(line)
                except ValueError:
                    continue

                evals_list = data.get("evals", [])
                if not evals_list:
                    continue

                main_eval = evals_list[0]
                if main_eval.get("depth", 0) < MIN_DEPTH:
                    continue

                pvs = main_eval.get("pvs", [])
                if not pvs:
                    continue

                raw_score = calc_score(pvs[0])

                fen = data["fen"]
                is_white_turn = " w " in fen

                if is_white_turn:
                    perspective_score = raw_score
                else:
                    perspective_score = -raw_score

                if perspective_score > MAX_CP_SCORE:
                    continue

                fen = data["fen"]
                piece_count = get_piece_count(fen)

                if piece_count > MAX_PIECES:
                    piece_count = MAX_PIECES
                if piece_count < MIN_PIECES:
                    piece_count = MIN_PIECES

                if bucket_counts[piece_count] >= SAMPLES_PER_BUCKET:
                    continue

                fen_hash = hash(fen)
                if fen_hash in seen_hashes:
                    continue
                seen_hashes.add(fen_hash)

                batch_fens.append(fen)
                batch_indices.append(piece_count)

                if len(batch_fens) >= BATCH_SIZE:
                    future = executor.submit(process_batch, list(batch_fens))
                    futures.append((future, list(batch_indices)))
                    batch_fens = []
                    batch_indices = []

                if len(futures) > 50:
                    pending = []
                    for fut, p_counts in futures:
                        if fut.done():
                            try:
                                results = fut.result()
                            except Exception as e:
                                print(f"Worker Error: {e}")
                                results = [None] * len(p_counts)

                            for i, packed_data in enumerate(results):
                                if packed_data is not None:
                                    p_count = p_counts[i]
                                    if bucket_counts[p_count] < SAMPLES_PER_BUCKET:
                                        boards_packed[current_samples] = packed_data
                                        bucket_counts[p_count] += 1
                                        current_samples += 1
                                        pbar_saved.update(1)

                                        if current_samples >= TRUE_TOTAL_SAMPLES:
                                            is_finished = True
                                            break

                            if is_finished:
                                break
                        else:
                            pending.append((fut, p_counts))
                    futures = pending

            if not is_finished and batch_fens:
                future = executor.submit(process_batch, batch_fens)
                futures.append((future, batch_indices))

            if not is_finished:
                for fut, p_counts in futures:
                    try:
                        results = fut.result()
                    except Exception:
                        continue

                    for i, packed_data in enumerate(results):
                        if packed_data is not None:
                            p_count = p_counts[i]
                            if bucket_counts[p_count] < SAMPLES_PER_BUCKET:
                                boards_packed[current_samples] = packed_data
                                bucket_counts[p_count] += 1
                                current_samples += 1
                                pbar_saved.update(1)
                                if current_samples >= TRUE_TOTAL_SAMPLES:
                                    is_finished = True
                                    break
                    if is_finished:
                        break

    # --- SAVE LOGIC (Now reachable) ---
    if current_samples < TRUE_TOTAL_SAMPLES:
        boards_packed = boards_packed[:current_samples]
        print(f"\nWarning: EOF reached. Collected {current_samples} boards.")
    else:
        print(f"\nTarget {TRUE_TOTAL_SAMPLES} reached!")

    print(f"Saving to {output_path}...")
    np.save(output_path, boards_packed)

    pbar_scan.close()
    pbar_saved.close()

    print("Done.")


if __name__ == "__main__":
    raw_file = DATA_PATH / "raw" / "lichess_db_eval.jsonl.zst"
    out_file = DATA_PATH / "processed" / "hard_negatives.npy"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    mine_hard_negatives_multiprocess(raw_file, out_file)
