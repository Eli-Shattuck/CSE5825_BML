import chess
import numpy as np


def board_to_packed(board):
    """Converts a chess.Board to an int8 array representation"""
    packed_array = [0] * 64
    is_black_turn = board.turn == chess.BLACK

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            is_friendly = piece.color == board.turn
            offset = 0 if is_friendly else 6
            val = piece.piece_type + offset
            target_square = (square ^ 63) if is_black_turn else square
            packed_array[target_square] = val

    return packed_array


def packed_to_board(packed_array):
    """Reconstructs a chess.Board from the int8 array"""
    board = chess.Board(None)
    board.clear()

    for sq, val in enumerate(packed_array):
        if val == 0:
            continue

        if 1 <= val <= 6:
            piece = chess.Piece(val, chess.WHITE)
        elif 7 <= val <= 12:
            piece = chess.Piece(val - 6, chess.BLACK)
        else:
            raise ValueError(f"Invalid piece value: {val}")

        board.set_piece_at(sq, piece)

    return board


def packed_to_tensor(packed_array):
    """Converts the packed int8 array to a (12, 8, 8) tensor representation"""

    tensor = np.zeros((12, 8, 8), dtype=np.int8)
    for sq, val in enumerate(packed_array):
        if val == 0:
            continue

        piece_index = val - 1
        row = sq // 8
        col = sq % 8
        tensor[piece_index, row, col] = 1

    return tensor


def tensor_to_board(tensor):
    """Reconstructs a chess.Board from the (12, 8, 8) tensor representation"""

    board = chess.Board(None)
    board.clear()

    for piece_index in range(12):
        positions = np.argwhere(tensor[piece_index] == 1)
        for pos in positions.T:
            row, col = pos
            sq = row * 8 + col
            if piece_index < 6:
                piece = chess.Piece(piece_index + 1, chess.WHITE)
            else:
                piece = chess.Piece(piece_index - 5, chess.BLACK)
            board.set_piece_at(sq, piece)

    return board
