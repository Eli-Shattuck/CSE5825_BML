import chess
import numpy as np


def board_to_packed(board: chess.Board):
    """
    Converts a chess.Board to an int8 array representation (Size 69).
    0-63: Square contents
    64-67: Castling Rights [Friendly-K, Friendly-Q, Enemy-K, Enemy-Q]
    68: En Passant Target Square (-1 if None)
    """
    packed_array = np.zeros(69, dtype=np.int8)

    is_black_turn = board.turn == chess.BLACK

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            is_friendly = piece.color == board.turn
            offset = 0 if is_friendly else 6
            val = piece.piece_type + offset

            target_square = (square ^ 63) if is_black_turn else square
            packed_array[target_square] = val

    us = board.turn
    them = not us

    packed_array[64] = int(board.has_kingside_castling_rights(us))
    packed_array[65] = int(board.has_queenside_castling_rights(us))
    packed_array[66] = int(board.has_kingside_castling_rights(them))
    packed_array[67] = int(board.has_queenside_castling_rights(them))

    if board.ep_square is not None and board.has_legal_en_passant():
        ep_target = (board.ep_square ^ 63) if is_black_turn else board.ep_square
        packed_array[68] = ep_target
    else:
        packed_array[68] = -1

    return packed_array


def packed_to_board(packed_array: np.ndarray) -> chess.Board:
    """Reconstructs a chess.Board from the int8 array"""
    board = chess.Board(None)
    board.clear()

    for sq, val in enumerate(packed_array[:64]):
        if val == 0:
            continue

        if 1 <= val <= 6:
            piece = chess.Piece(val, chess.WHITE)
        elif 7 <= val <= 12:
            piece = chess.Piece(val - 6, chess.BLACK)
        else:
            raise ValueError(f"Invalid piece value: {val} at square {sq}")

        board.set_piece_at(sq, piece)

    castling_mask = chess.BB_EMPTY

    if packed_array[64]:
        castling_mask |= chess.BB_H1  # White King-side
    if packed_array[65]:
        castling_mask |= chess.BB_A1  # White Queen-side
    if packed_array[66]:
        castling_mask |= chess.BB_H8  # Black King-side
    if packed_array[67]:
        castling_mask |= chess.BB_A8  # Black Queen-side

    board.castling_rights = castling_mask

    ep_val = packed_array[68]
    if ep_val != -1:
        board.ep_square = int(ep_val)
    else:
        board.ep_square = None

    return board


def packed_to_tensor(packed_array):
    """
    Inflates packed array (69,) into Tensor (17, 8, 8).
    """
    tensor = np.zeros((17, 8, 8), dtype=np.float32)

    board_data = packed_array[:64]
    for sq, val in enumerate(board_data):
        if val > 0:
            plane_idx = val - 1
            row, col = divmod(sq, 8)
            tensor[plane_idx, row, col] = 1.0

    castling_data = packed_array[64:68]
    for i, has_right in enumerate(castling_data):
        if has_right:
            tensor[12 + i, :, :] = 1.0

    ep_sq = packed_array[68]
    if ep_sq != -1:
        row, col = divmod(ep_sq, 8)
        tensor[16, row, col] = 1.0

    return tensor


def tensor_to_board(tensor):
    """Reconstructs a chess.Board from the (17, 8, 8) tensor representation"""

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

    castling_rights = chess.BB_EMPTY

    if np.any(tensor[12] == 1):
        castling_rights |= chess.BB_H1  # White King-side
    if np.any(tensor[13] == 1):
        castling_rights |= chess.BB_A1  # White Queen-side
    if np.any(tensor[14] == 1):
        castling_rights |= chess.BB_H8  # Black King-side
    if np.any(tensor[15] == 1):
        castling_rights |= chess.BB_A8  # Black Queen-side

    board.castling_rights = castling_rights

    ep_positions = np.argwhere(tensor[16] == 1)
    if ep_positions.size > 0:
        row, col = ep_positions[0]
        ep_sq = row * 8 + col
        board.ep_square = ep_sq
    else:
        board.ep_square = None

    return board


def clean_state_dict(state_dict):
    """
    Removes '_orig_mod.' prefixes from state dict keys.
    """
    cleaned_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "")
        cleaned_dict[new_key] = value
    return cleaned_dict
