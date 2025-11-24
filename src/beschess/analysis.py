import chess
import chess.pgn
import chess.engine
import os
from stockfish import Stockfish as LegacyStockfish
from typing import Generator, Tuple
from dataclasses import dataclass
import logging


@dataclass
class StockFishConfig:
    depth: int | None = None
    nodes: int | None = None
    threads: int = 8


class StockFish:
    instance: chess.engine.SimpleEngine
    config: StockFishConfig

    def __init__(self, config: StockFishConfig = StockFishConfig()):
        self.config = config
        stockfish_path = os.getenv("STOCKFISH_PATH") or "/usr/local/bin/stockfish"
        self.instance = chess.engine.SimpleEngine.popen_uci(stockfish_path)

        if "Use NNUE" in self.instance.options:
            self.instance.configure({"Use NNUE": True})

        self.instance.configure({"Threads": self.config.threads})

    def quit(self):
        self.instance.quit()


def is_puzzle(
    stockfish: StockFish,
    board: chess.Board,
    amax_cp: int,
    amax_cp_diff: int,
):
    if stockfish.config.depth is None or stockfish.config.nodes is None:
        raise ValueError("Either depth or nodes must be specified in StockFishConfig.")

    if stockfish.config.depth is not None:
        limit = chess.engine.Limit(depth=stockfish.config.depth)
    elif stockfish.config.nodes > 0:
        limit = chess.engine.Limit(nodes=stockfish.config.nodes)
    else:
        raise ValueError("Either depth or nodes must be greater than zero.")

    info = stockfish.instance.analyse(
        board,
        limit,
        multipv=2,
    )

    if len(info) < 2:
        return False

    s1 = info[0]["score"].relative.score(mate_score=10000)
    s2 = info[1]["score"].relative.score(mate_score=10000)

    logging.info(f"Score 1: {s1}, Score 2: {s2}")

    return abs(s1) >= amax_cp and abs(s2 - s1) >= amax_cp_diff


# --- LEGACY STOCKFISH ANALYSIS FUNCTION ---


def engine_analysis(
    game: chess.pgn.Game,
    depth: int = 12,
) -> Generator[Tuple[Tuple[str, str], int, chess.Move, int]]:
    # --- 1. INITIALIZE STOCKFISH ---
    try:
        stockfish = LegacyStockfish(
            path=os.getenv("STOCKFISH_PATH") or "/usr/local/bin/stockfish",
            depth=depth,
        )
        stockfish.update_engine_parameters({"Threads": 8})
    except FileNotFoundError:
        print("ERROR: Stockfish engine not found. Please update the path.")
        exit()

    board = game.board()
    white_player = game.headers.get("White", "Unknown")
    black_player = game.headers.get("Black", "Unknown")
    print(f"Analyzing: {game.headers['White']} vs {game.headers['Black']}\n")

    for number, move in enumerate(game.mainline_moves()):
        # Set the board to the position *before* the move
        stockfish.set_fen_position(board.fen())

        # Get engine's evaluation in centipawns
        # Positive score = White's advantage
        # Negative score = Black's advantage
        evaluation_cp = stockfish.get_evaluation()["value"]

        # Normalize score to White's perspective
        # if board.turn == chess.BLACK:
        #     evaluation_cp = -evaluation_cp

        # Get the best move suggested by the engine
        best_move_san = board.san(chess.Move.from_uci(stockfish.get_best_move()))

        # Get the move number and player turn
        move_number = (number // 2) + 1
        player = "White" if board.turn == chess.WHITE else "Black"

        yield (white_player, black_player), number, move, evaluation_cp

        # Make the move on the board to advance to the next position
        board.push(move)

    print("Analysis complete.")
