import chess
import pytest
import numpy as np

from beschess.utils import (
    board_to_packed,
    packed_to_board,
    tensor_to_board,
    packed_to_tensor,
)


# --- Import your corrected functions here or ensure they are in the same file ---
# from chess_utils import board_to_packed, packed_to_board, packed_to_tensor, tensor_to_board


# Helper for debugging/assertions
def get_canonical_fen(board: chess.Board) -> str:
    """
    Manually applies the 180-degree rotation and color swap
    to predict what the 'tensor' board should look like.
    """
    if board.turn == chess.WHITE:
        return board.fen()

    # If Black to move, we simulate the transformation manually:
    # 1. Rotate 180 degrees (sq ^ 63)
    # 2. Swap colors (Friendly -> White, Enemy -> Black)
    new_board = chess.Board(None)
    new_board.turn = chess.WHITE
    new_board.castling_rights = chess.BB_EMPTY

    # 1. Transfer Pieces
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            # Rotate square
            target_sq = sq ^ 63
            # Swap color logic: If it matched the turn, it becomes White
            new_color = chess.WHITE if piece.color == board.turn else chess.BLACK
            new_board.set_piece_at(target_sq, chess.Piece(piece.piece_type, new_color))

    # 2. Transfer Castling Rights
    # Us (Black) -> becomes White rights
    # Them (White) -> becomes Black rights
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

    # 3. Transfer En Passant
    if board.ep_square is not None:
        new_board.ep_square = board.ep_square ^ 63

    return new_board.fen()


# ==========================================
# TESTS
# ==========================================


def test_starting_position_consistency():
    """Test that the starting board remains identical (since it's White's turn)."""
    board = chess.Board()

    # 1. To Packed
    packed = board_to_packed(board)
    assert packed.shape == (69,)
    assert packed.dtype == np.int8

    # 2. Packed -> Board
    recon_board_1 = packed_to_board(packed)
    assert recon_board_1.fen() == board.fen()

    # 3. Packed -> Tensor
    tensor = packed_to_tensor(packed)
    assert tensor.shape == (17, 8, 8)

    # 4. Tensor -> Board
    recon_board_2 = tensor_to_board(tensor)
    assert recon_board_2.fen() == board.fen()


def test_black_perspective_rotation():
    """
    CRITICAL TEST:
    Verifies that if it is Black's turn, the board is rotated 180 degrees
    and colors are swapped so that the output is White-to-move.
    """
    board = chess.Board(None)
    board.turn = chess.BLACK

    # Setup: Black King on E8, White King on E1.
    # Add a Black Pawn on A7.
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.A7, chess.Piece(chess.PAWN, chess.BLACK))

    # Expected Output (Canonical):
    # Active player (Black) becomes White.
    # Rotation: E8 -> E1, E1 -> E8, A7 -> H2.
    # Result: White King E1, Black King E8, White Pawn H2.

    packed = board_to_packed(board)
    recon_board = packed_to_board(packed)

    assert recon_board.turn == chess.WHITE
    assert recon_board.piece_at(chess.E1) == chess.Piece(
        chess.KING, chess.WHITE
    )  # Was Black King E8
    assert recon_board.piece_at(chess.E8) == chess.Piece(
        chess.KING, chess.BLACK
    )  # Was White King E1
    assert recon_board.piece_at(chess.H2) == chess.Piece(
        chess.PAWN, chess.WHITE
    )  # Was Black Pawn A7
    assert recon_board.piece_at(chess.A7) is None


def test_castling_rights_preservation():
    """Test that castling rights are preserved and mapped correctly."""
    board = chess.Board()
    # Remove White Queenside and Black Kingside rights
    board.castling_rights = chess.BB_H1 | chess.BB_A8

    packed = board_to_packed(board)
    recon = packed_to_board(packed)

    assert recon.has_kingside_castling_rights(chess.WHITE)
    assert not recon.has_queenside_castling_rights(chess.WHITE)
    assert not recon.has_kingside_castling_rights(chess.BLACK)
    assert recon.has_queenside_castling_rights(chess.BLACK)

    # Test via Tensor path
    tensor = packed_to_tensor(packed)
    recon_2 = tensor_to_board(tensor)
    assert recon_2.castling_rights == recon.castling_rights


def test_en_passant_tracking():
    """Test that En Passant squares are tracked and rotated correctly."""
    board = chess.Board()
    board.clear()
    board.turn = chess.BLACK

    # Black just moved pawn to E5, White can EP at E6.
    # BUT wait, the input is 'board'.
    # Let's say it's Black's turn. White just moved a pawn 2 squares.
    # White Pawn on C4. EP Target is C3.

    board.turn = chess.BLACK
    board.ep_square = chess.C3
    # We must have a pawn on C4 or similar for it to be valid,
    # but the serializer mainly cares that the flag is set.
    # To be "legal" in chess logic, there must be a pawn.
    # board.has_legal_en_passant() checks for attackers,
    # so we need to set up a scenario where EP is actually valid.

    # Scenario: White Pawn on C4 (just moved c2-c4). Black Pawn on D4.
    board.set_piece_at(chess.C4, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.D4, chess.Piece(chess.PAWN, chess.BLACK))
    board.ep_square = chess.C3

    assert board.has_legal_en_passant()

    # EXPECTED Rotation:
    # Black is active -> Becomes White.
    # EP Square C3 (18) -> Rotates 180 -> F6 (45).

    packed = board_to_packed(board)
    recon = packed_to_board(packed)

    assert recon.ep_square == chess.F6

    # Check Tensor path
    tensor = packed_to_tensor(packed)
    recon_2 = tensor_to_board(tensor)
    assert recon_2.ep_square == chess.F6


def test_empty_board():
    """Ensure no crashes on empty board (though technically invalid state)."""
    board = chess.Board(None)
    board.clear()

    packed = board_to_packed(board)
    assert np.all(packed == 0) or np.all(
        packed[:64] == 0
    )  # Castling/EP might be -1 or 0

    tensor = packed_to_tensor(packed)
    assert np.sum(tensor) == 0

    recon = tensor_to_board(tensor)
    assert recon.piece_map() == {}


def test_random_positions_consistency():
    """
    Fuzzing test: Generates random legal games and checks if
    Original -> Packed -> Tensor -> Board
    is equivalent to the Canonical Prediction.
    """
    import random

    for _ in range(20):  # Run 20 random games
        board = chess.Board()
        moves_to_play = random.randint(5, 40)

        for _ in range(moves_to_play):
            if board.is_game_over():
                break
            move = random.choice(list(board.legal_moves))
            board.push(move)

        # Calculate what the canonical string should be
        expected_fen_parts = get_canonical_fen(board).split(" ")
        # We only care about piece placement (0), active color (1), rights (2), ep (3)
        # Halfmove/Fullmove clocks are NOT preserved by this compression
        expected_core = " ".join(expected_fen_parts[:4])

        # Run conversion
        packed = board_to_packed(board)
        tensor = packed_to_tensor(packed)
        recon_board = tensor_to_board(tensor)

        recon_fen_parts = recon_board.fen().split(" ")
        recon_core = " ".join(recon_fen_parts[:4])

        assert recon_core == expected_core, (
            f"Failed on FEN: {board.fen()}\nExpected: {expected_core}\nGot: {recon_core}"
        )
