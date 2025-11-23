import chess
import pytest

from beschess.analysis import is_puzzle, StockFish


@pytest.fixture(scope="session")
def stockfish():
    stockfish = StockFish()
    yield stockfish
    stockfish.quit()


@pytest.fixture
def puzzles():
    return [
        "8/pp3pBp/3k4/3Pb3/2K5/5PP1/7P/8 w - - 0 1",
        "r4B2/pp4bk/2n1qpb1/8/2p3N1/2Q5/PP3PP1/R5K1 w - - 0 1",
        "8/6p1/5pkp/1R2r3/6KP/5P2/6P1/8 w - - 0 1",
        "1k2r2r/ppp1qppp/5n2/1B3P2/1R6/2Q1P1P1/2P1K2P/7R w - - 0 1",
        "1k5r/pp6/8/1PR1p3/PB1pPqPp/4bP2/2Q3K1/8 w - - 0 1",
        "r7/1pk3pp/4pp2/2bp4/q4P2/2B1P2P/1R1QK1P1/rR6 w - - 0 1",
        "2r1rk2/BQpbq1pp/4pn2/4Rp2/8/8/PPP2PPP/3R1K2 w - - 0 1",
        "1kr3r1/pp1n1ppp/3q1n2/3p4/2p2P2/1P2P1Q1/PBPPB2P/1KR4R w - - 0 1",
        "8/8/1p1k1p2/1p1Pp2p/1P2Pp2/4K1PP/8/8 w - - 0 1",
        "8/5p2/Q5p1/1p3k2/P5qp/8/5P1K/8 w - - 0 1",
    ]


@pytest.fixture
def non_puzzles():
    return [
        # Starting position
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        # Opening positions
        "rnbqkb1r/ppp1pp1p/6p1/3n4/3P4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 1",
        # Positions taken from high-level draw games
        "r3kb1r/1q3p1p/p2p1np1/P2Pp3/RpB5/1N1Q1P2/1PP3PP/4K2R w Kk - 8 20",
        "8/5pk1/4p1p1/p1b4p/8/1P1BPKP1/5P1P/8 b - - 8 34",
        "5rk1/3q1p2/p1Np1npb/P2Pp2p/R1B5/5P2/1rP1Q1PP/5RK1 b - - 1 24",
        "4k3/R7/8/3KP3/8/6r1/8/8 b - - 0 1",
        # Crushing positions that are not puzzles
        "8/6k1/8/6K1/8/6P1/8/8 w - - 0 1",
    ]


def test_is_puzzle(stockfish, puzzles, caplog):
    caplog.set_level("INFO")
    amax_cp = 50
    amax_cp_diff = 100
    for fen in puzzles:
        board = chess.Board(fen)
        assert is_puzzle(stockfish, board, amax_cp, amax_cp_diff)


def test_is_not_puzzle(stockfish, non_puzzles, caplog):
    caplog.set_level("INFO")
    amax_cp = 50
    amax_cp_diff = 100
    for fen in non_puzzles:
        board = chess.Board(fen)
        assert not is_puzzle(stockfish, board, amax_cp, amax_cp_diff)
