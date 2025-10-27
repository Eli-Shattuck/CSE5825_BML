import chess
import chess.pgn
import os
from stockfish import Stockfish
from typing import Generator, Tuple


def engine_analysis(
    game: chess.pgn.Game,
    depth: int = 12,
) -> Generator[Tuple[Tuple[str, str], int, chess.Move, int]]:
    # --- 1. INITIALIZE STOCKFISH ---
    try:
        stockfish = Stockfish(
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
