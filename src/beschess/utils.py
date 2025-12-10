import chess
import numpy as np


def board_to_packed(board: chess.Board) -> np.ndarray:
    """
    Converts a chess.Board to an int8 array representation (Size 133).
    0-63: Square contents
        - first 4 bits: piece type (1-6 for friendly, 7-12 for enemy, 0 for empty)
        - next 1 bit: can be legally moved to
        - last 3 bits: unused
    64-127: Square Attack Info
        - first 4 bits: n-attacking pieces (friendly)
        - last 4 bits: n-attacking pieces (enemy)
    128-131: Castling Rights [Friendly-K, Friendly-Q, Enemy-K, Enemy-Q]
    132: En Passant Target Square (0-63), 255 if none
    """
    packed_array = np.zeros(133, dtype=np.uint8)

    is_black_turn = board.turn == chess.BLACK

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            is_friendly = piece.color == board.turn
            offset = 0 if is_friendly else 6
            val = piece.piece_type + offset

            target_square = (square ^ 63) if is_black_turn else square
            packed_array[target_square] = val

        board_friendly_attacks = len(board.attackers(board.turn, square))
        board_enemy_attacks = len(board.attackers(not board.turn, square))

        target_square = (square ^ 63) if is_black_turn else square
        packed_array[target_square + 64] |= board_friendly_attacks & 0b00001111
        packed_array[target_square + 64] |= (board_enemy_attacks & 0b00001111) << 4

    for move in board.legal_moves:
        to_sq = move.to_square

        target_to_sq = (to_sq ^ 63) if is_black_turn else to_sq
        packed_array[target_to_sq] |= 0b00010000

    us = board.turn
    them = not us

    packed_array[128] = int(board.has_kingside_castling_rights(us))
    packed_array[129] = int(board.has_queenside_castling_rights(us))
    packed_array[130] = int(board.has_kingside_castling_rights(them))
    packed_array[131] = int(board.has_queenside_castling_rights(them))

    if board.ep_square is not None and board.has_legal_en_passant():
        ep_target = (board.ep_square ^ 63) if is_black_turn else board.ep_square
        packed_array[132] = ep_target
    else:
        packed_array[132] = 255

    return packed_array


def packed_to_board(packed_array: np.ndarray) -> chess.Board:
    """Reconstructs a chess.Board from the new int8 array (Size 133)."""
    board = chess.Board(None)
    board.clear()

    for sq, raw_val in enumerate(packed_array[:64]):
        val = raw_val & 0x0F

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

    if packed_array[128]:
        castling_mask |= chess.BB_H1  # White King-side
    if packed_array[129]:
        castling_mask |= chess.BB_A1  # White Queen-side
    if packed_array[130]:
        castling_mask |= chess.BB_H8  # Black King-side
    if packed_array[131]:
        castling_mask |= chess.BB_A8  # Black Queen-side

    board.castling_rights = castling_mask

    ep_val = packed_array[132]
    if ep_val != 255:
        board.ep_square = int(ep_val)
    else:
        board.ep_square = None

    board.turn = chess.WHITE

    return board


def packed_to_tensor(packed_array: np.ndarray) -> np.ndarray:
    """
    Inflates packed array (133,) into Tensor (20, 8, 8).
    Channels 0-16: Board State
    Channels 17-19: Geometric Features (Attacks/Legal Moves)
    """
    tensor = np.zeros((20, 8, 8), dtype=np.float32)

    board_data = packed_array[:64]

    for sq, raw_val in enumerate(board_data):
        row, col = divmod(sq, 8)

        val = raw_val & 0x0F
        if val > 0:
            plane_idx = val - 1
            tensor[plane_idx, row, col] = 1.0

        if raw_val & 0x10:
            tensor[19, row, col] = 1.0

    attack_data = packed_array[64:128]

    for sq, val in enumerate(attack_data):
        row, col = divmod(sq, 8)

        my_attacks = val & 0x0F
        if my_attacks > 0:
            tensor[17, row, col] = min(my_attacks / 5.0, 1.0)

        enemy_attacks = (val >> 4) & 0x0F
        if enemy_attacks > 0:
            tensor[18, row, col] = min(enemy_attacks / 5.0, 1.0)

    castling_data = packed_array[128:132]
    for i, has_right in enumerate(castling_data):
        if has_right:
            tensor[12 + i, :, :] = 1.0

    ep_sq = packed_array[132]
    if ep_sq != 255:
        row, col = divmod(ep_sq, 8)
        tensor[16, row, col] = 1.0

    return tensor


def tensor_to_board(tensor: np.ndarray) -> chess.Board:
    """Reconstructs a chess.Board from the tensor representation.
    Ignores geometric channels (17-19) as they are derived properties."""

    board = chess.Board(None)
    board.clear()
    board.turn = chess.WHITE

    for piece_index in range(12):
        positions = np.argwhere(tensor[piece_index] == 1)

        for pos in positions:
            row, col = pos
            sq = row * 8 + col

            if piece_index < 6:
                piece = chess.Piece(piece_index + 1, chess.WHITE)
            else:
                piece = chess.Piece(piece_index - 5, chess.BLACK)

            board.set_piece_at(sq, piece)

    castling_rights = chess.BB_EMPTY
    if (tensor[12] == 1).any():
        castling_rights |= chess.BB_H1
    if (tensor[13] == 1).any():
        castling_rights |= chess.BB_A1
    if (tensor[14] == 1).any():
        castling_rights |= chess.BB_H8
    if (tensor[15] == 1).any():
        castling_rights |= chess.BB_A8

    board.castling_rights = castling_rights

    enpassant_sq = (tensor[16] == 1).nonzero()

    if len(enpassant_sq[0]) > 0:
        ep_rows, ep_cols = enpassant_sq
        ep_sq = ep_rows[0] * 8 + ep_cols[0]
        board.ep_square = int(ep_sq)
    else:
        board.ep_square = None

    return board


# def packed_to_board(packed_array: np.ndarray) -> chess.Board:
#     """Reconstructs a chess.Board from the int8 array"""
#     board = chess.Board(None)
#     board.clear()
#
#     for sq, val in enumerate(packed_array[:64]):
#         if val == 0:
#             continue
#
#         if 1 <= val <= 6:
#             piece = chess.Piece(val, chess.WHITE)
#         elif 7 <= val <= 12:
#             piece = chess.Piece(val - 6, chess.BLACK)
#         else:
#             raise ValueError(f"Invalid piece value: {val} at square {sq}")
#
#         board.set_piece_at(sq, piece)
#
#     castling_mask = chess.BB_EMPTY
#
#     if packed_array[64]:
#         castling_mask |= chess.BB_H1  # White King-side
#     if packed_array[65]:
#         castling_mask |= chess.BB_A1  # White Queen-side
#     if packed_array[66]:
#         castling_mask |= chess.BB_H8  # Black King-side
#     if packed_array[67]:
#         castling_mask |= chess.BB_A8  # Black Queen-side
#
#     board.castling_rights = castling_mask
#
#     ep_val = packed_array[68]
#     if ep_val != -1:
#         board.ep_square = int(ep_val)
#     else:
#         board.ep_square = None
#
#     return board


# def packed_to_tensor(packed_array: np.ndarray) -> np.ndarray:
#     """
#     Inflates packed array (69,) into Tensor (17, 8, 8).
#     """
#     tensor = np.zeros((17, 8, 8), dtype=np.float32)
#
#     board_data = packed_array[:64]
#     for sq, val in enumerate(board_data):
#         if val > 0:
#             plane_idx = val - 1
#             row, col = divmod(sq, 8)
#             tensor[plane_idx, row, col] = 1.0
#
#     castling_data = packed_array[64:68]
#     for i, has_right in enumerate(castling_data):
#         if has_right:
#             tensor[12 + i, :, :] = 1.0
#
#     ep_sq = packed_array[68]
#     if ep_sq != -1:
#         row, col = divmod(ep_sq, 8)
#         tensor[16, row, col] = 1.0
#
#     return tensor


# def tensor_to_board(tensor: np.ndarray) -> chess.Board:
#     """Reconstructs a chess.Board from the (17, 8, 8) tensor representation"""
#     board = chess.Board(None)
#     board.clear()
#     board.turn = chess.WHITE
#
#     for piece_index in range(12):
#         positions = np.argwhere(tensor[piece_index] == 1)
#
#         for pos in positions:
#             row, col = pos
#             sq = row * 8 + col
#
#             if piece_index < 6:
#                 piece = chess.Piece(piece_index + 1, chess.WHITE)
#             else:
#                 piece = chess.Piece(piece_index - 5, chess.BLACK)
#
#             board.set_piece_at(sq, piece)
#
#     castling_rights = chess.BB_EMPTY
#     if (tensor[12] == 1).any():
#         castling_rights |= chess.BB_H1
#     if (tensor[13] == 1).any():
#         castling_rights |= chess.BB_A1
#     if (tensor[14] == 1).any():
#         castling_rights |= chess.BB_H8
#     if (tensor[15] == 1).any():
#         castling_rights |= chess.BB_A8
#
#     board.castling_rights = castling_rights
#
#     enpassant_sq = (tensor[16] == 1).nonzero()
#
#     if len(enpassant_sq[0]) > 0:
#         ep_rows, ep_cols = enpassant_sq
#         ep_sq = ep_rows[0] * 8 + ep_cols[0]
#         board.ep_square = int(ep_sq)
#     else:
#         board.ep_square = None
#
#     return board


def clean_state_dict(state_dict):
    """
    Removes '_orig_mod.' prefixes from state dict keys.
    """
    cleaned_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "")
        cleaned_dict[new_key] = value
    return cleaned_dict
