import sys
from pathlib import Path

import chess
import chess.pgn
import numpy as np
from pandas.io.pytables import DataCol
from tqdm import tqdm
from beschess.packed import board_to_packed

from beschess.analysis import StockFish, StockFishConfig, is_puzzle
from beschess.load.load_game import load_game_zstd

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

DATA_PATH = Path(__file__).resolve().parent.parent / "data"

TARGET_SAMPLES = 1_000_000
# TARGET_SAMPLES = 10

MIN_ELO = 1501
MAX_EVAL_CP = 50
MAX_DIFF_CP = 30
MIN_ply = 30
MAX_ply = 80


def mine_quiet_boards(pgn_path, output_path, total=None):
    try:
        stockfish = StockFish(StockFishConfig(depth=12, threads=16))

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

            # Filter 1: Only look at drawn/close games to save time
            if game.headers.get("Result") not in ["1/2-1/2", "*"]:
                continue

            board = game.board()

            # Iterate through moves
            for ply, move in enumerate(game.mainline_moves()):
                board.push(move)

                # Filter 2: Window & Checks
                if ply < MIN_ply or ply > MAX_ply:
                    continue
                if board.is_check():
                    continue

                # # Filter 3: Random Sampling (Don't take every move from one game)
                if np.random.random() > 0.1:
                    continue

                # Filter 4: Engine Analysis (The Heavy Lift)
                try:
                    if is_puzzle(
                        stockfish,
                        board,
                        amax_cp=MAX_EVAL_CP,
                        amax_cp_diff=MAX_DIFF_CP,
                    ):
                        continue

                    boards_packed[boards_sampled] = board_to_packed(board)

                    puzzle_bar.update(1)
                    boards_sampled += 1

                    break

                except Exception as e:
                    continue

        stockfish.quit()
        print(f"Mined {boards_sampled} quiet boards.")
        np.save(output_path, boards_packed[:boards_sampled])
    except Exception as e:
        stockfish.quit()
        raise e


if __name__ == "__main__":
    mine_quiet_boards(
        DATA_PATH / "raw" / "lichess_db_standard_rated_2013-01.pgn.zst",
        DATA_PATH / "processed" / "quiet_2013-01_boards_packed.npy",
        121_332,
    )

    # mine_quiet_boards(
    #     DATA_PATH / "raw" / "lichess_db_standard_rated_2019-01.pgn.zst",
    #     DATA_PATH / "processed" / "quiet_2019-01_boards_packed.npy",
    #     33_886_899,
    # )
