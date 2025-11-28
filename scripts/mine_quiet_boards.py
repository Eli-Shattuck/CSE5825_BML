from pathlib import Path

import numpy as np
from tqdm import tqdm

from beschess.analysis import StockFish, StockFishConfig, is_puzzle
from beschess.load.load_game import load_game_zstd
from beschess.utils import board_to_packed

DATA_PATH = Path(__file__).resolve().parent.parent / "data"

TARGET_SAMPLES = 1_000_000

MIN_ELO = 1501
MAX_DIFF_CP = 30
MIN_ply = 30
MAX_ply = 80


def mine_quiet_boards(pgn_path, output_path, total=None):
    try:
        stockfish = StockFish(StockFishConfig(nodes=5000, threads=16))

        boards_packed = np.zeros((TARGET_SAMPLES, 64), dtype=np.int8)
        boards_sampled = 0

        print("Opening PGN...")
        games = load_game_zstd(pgn_path)

        puzzle_bar = tqdm(total=TARGET_SAMPLES)
        if total:
            total_bar = tqdm(total=total)

        while boards_sampled < TARGET_SAMPLES:
            game = next(games, None)
            total_bar.update(1)

            if game is None:
                print("out of games")
                break

            white = game.headers.get("WhiteElo")
            black = game.headers.get("BlackElo")

            if white is None or black is None:
                continue

            if not white.isnumeric() or not black.isnumeric():
                continue

            if int(white) < MIN_ELO or int(black) < MIN_ELO:
                continue

            event = game.headers.get("Event", "")
            if "Blitz" in event or "Bullet" in event:
                continue

            # WARN:
            if game.headers.get("Result") not in ["1/2-1/2", "*"]:
                continue

            board = game.board()

            for ply, move in enumerate(game.mainline_moves()):
                board.push(move)

                # Window & Checks
                if ply < MIN_ply or ply > MAX_ply:
                    continue
                if board.is_check():
                    continue

                # Random Sampling
                if np.random.random() > 0.1:
                    continue

                # Engine Analysis
                try:
                    if is_puzzle(
                        stockfish,
                        board,
                        amax_cp_diff=MAX_DIFF_CP,
                    ):
                        continue

                    boards_packed[boards_sampled] = board_to_packed(board)

                    puzzle_bar.update(1)
                    boards_sampled += 1

                    break

                except Exception as e:
                    print(f"Engine analysis error: {e}")
                    continue

        stockfish.quit()
        print(f"Mined {boards_sampled} quiet boards.")
        np.save(output_path, boards_packed[:boards_sampled])
        puzzle_bar.close()
    except Exception as e:
        stockfish.quit()
        raise e


if __name__ == "__main__":
    mine_quiet_boards(
        DATA_PATH / "raw" / "lichess_db_standard_rated_2013-01.pgn.zst",
        DATA_PATH / "processed" / "quiet_2013-01_boards_packed.npy",
        121_332,
    )
