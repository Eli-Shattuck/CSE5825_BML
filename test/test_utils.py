import chess
import numpy as np
import pytest

from beschess.utils import (
    board_to_packed,
    packed_to_board,
    tensor_to_board,
    packed_to_tensor,
)


def get_canonical_fen(board: chess.Board) -> str:
    """
    Manually applies the 180-degree rotation and color swap
    to predict what the 'tensor' board should look like.
    """
    if board.turn == chess.WHITE:
        return board.fen()

    new_board = chess.Board(None)
    new_board.turn = chess.WHITE
    new_board.castling_rights = chess.BB_EMPTY

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            target_sq = sq ^ 63
            new_color = chess.WHITE if piece.color == board.turn else chess.BLACK
            new_board.set_piece_at(target_sq, chess.Piece(piece.piece_type, new_color))

    us = board.turn
    them = not us
    if board.has_kingside_castling_rights(us):
        new_board.castling_rights |= chess.BB_H1
    if board.has_queenside_castling_rights(us):
        new_board.castling_rights |= chess.BB_A1
    if board.has_kingside_castling_rights(them):
        new_board.castling_rights |= chess.BB_H8
    if board.has_queenside_castling_rights(them):
        new_board.castling_rights |= chess.BB_A8

    if board.ep_square is not None:
        new_board.ep_square = board.ep_square ^ 63

    return new_board.fen()


def test_starting_position_consistency():
    """Test that the starting board remains identical (since it's White's turn)."""
    board = chess.Board()

    packed = board_to_packed(board)
    assert packed.shape == (133,)
    assert packed.dtype == np.uint8

    recon_board_1 = packed_to_board(packed)
    assert recon_board_1.fen() == board.fen()

    tensor = packed_to_tensor(packed)
    assert tensor.shape == (20, 8, 8)

    recon_board_2 = tensor_to_board(tensor)
    assert recon_board_2.fen() == board.fen()


def test_black_perspective_rotation():
    """
    Verifies that if it is Black's turn, the board is rotated 180 degrees
    and colors are swapped so that the output is White-to-move.
    """
    board = chess.Board(None)
    board.turn = chess.BLACK

    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.A7, chess.Piece(chess.PAWN, chess.BLACK))

    packed = board_to_packed(board)
    recon_board = packed_to_board(packed)

    assert recon_board.turn == chess.WHITE
    assert recon_board.piece_at(chess.D1) == chess.Piece(chess.KING, chess.WHITE)
    assert recon_board.piece_at(chess.D8) == chess.Piece(chess.KING, chess.BLACK)
    assert recon_board.piece_at(chess.H2) == chess.Piece(chess.PAWN, chess.WHITE)
    assert recon_board.piece_at(chess.A7) is None


def test_castling_rights_preservation():
    """Test that castling rights are preserved and mapped correctly."""
    board = chess.Board()
    board.castling_rights = chess.BB_H1 | chess.BB_A8

    packed = board_to_packed(board)
    recon = packed_to_board(packed)

    assert recon.has_kingside_castling_rights(chess.WHITE)
    assert not recon.has_queenside_castling_rights(chess.WHITE)
    assert not recon.has_kingside_castling_rights(chess.BLACK)
    assert recon.has_queenside_castling_rights(chess.BLACK)

    tensor = packed_to_tensor(packed)
    recon_2 = tensor_to_board(tensor)
    assert recon_2.castling_rights == recon.castling_rights


def test_en_passant_tracking():
    """Test that En Passant squares are tracked and rotated correctly."""
    board = chess.Board()
    board.clear()
    board.turn = chess.BLACK

    board.set_piece_at(chess.C4, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.D4, chess.Piece(chess.PAWN, chess.BLACK))

    b = chess.Board()
    b.push_san("e4")

    b = chess.Board("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2")
    b.set_piece_at(chess.D5, chess.Piece(chess.PAWN, chess.WHITE))
    assert b.has_legal_en_passant()

    packed = board_to_packed(b)
    recon = packed_to_board(packed)
    assert recon.ep_square == chess.C6


def test_geometry_channels():
    """
    Verify the geometric channels (17, 18, 19) are populated.
    """
    board = chess.Board()

    packed = board_to_packed(board)
    tensor = packed_to_tensor(packed)

    assert tensor[17, 2, 3] > 0
    assert tensor[17, 2, 5] > 0

    assert tensor[19, 2, 5] == 1.0  # F3
    assert tensor[19, 2, 7] == 1.0  # H3

    assert tensor[18, 5, 3] > 0
    assert tensor[18, 5, 5] > 0


def test_random_positions_consistency():
    """
    Fuzzing test: Generates random legal games.
    """
    import random

    random.seed(42)

    for _ in range(20):
        board = chess.Board()
        moves_to_play = random.randint(5, 40)

        for _ in range(moves_to_play):
            if board.is_game_over():
                break
            move = random.choice(list(board.legal_moves))
            board.push(move)

        expected_fen_parts = get_canonical_fen(board).split(" ")
        expected_core = " ".join(expected_fen_parts[:4])

        packed = board_to_packed(board)
        tensor = packed_to_tensor(packed)
        recon_board = tensor_to_board(tensor)

        recon_fen_parts = recon_board.fen().split(" ")
        recon_core = " ".join(recon_fen_parts[:4])

        assert recon_core == expected_core, (
            f"Failed on FEN: {board.fen()}\nExpected: {expected_core}\nGot: {recon_core}"
        )
