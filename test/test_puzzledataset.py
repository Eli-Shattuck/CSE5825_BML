import numpy as np
import pytest
import torch

from beschess.dataset.embedding import PuzzleDataset


def test_puzzle_dataset():
    quiet_boards = np.array(
        [
            np.eye(8, dtype=np.uint8).flatten(),
            np.fliplr(np.eye(8, dtype=np.uint8)).flatten(),
        ]
    )
    puzzle_boards = np.array(
        [
            np.repeat(np.arange(8, dtype=np.uint8), 8),
            np.tile(np.arange(1, 9, dtype=np.uint8), 8),
        ]
    )
    puzzle_labels = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    dataset = PuzzleDataset(quiet_boards, puzzle_boards, puzzle_labels)
    assert len(dataset) == 4  # 2 quiet + 2 puzzles
    for i in range(len(dataset)):
        board_tensor, label = dataset[i]
        assert board_tensor.shape == (12, 8, 8)
        assert label.shape == (16,)

    # Check first two are quiet boards
    board_tensor, label = dataset[0]
    assert label[0] == 1.0
    assert board_tensor.sum().item() == 8

    board_tensor, label = dataset[1]
    assert label[0] == 1.0
    assert board_tensor.sum().item() == 8

    # Check last two are puzzle boards
    board_tensor, label = dataset[2]
    assert label[1] == 1.0
    assert board_tensor.sum().item() == 56

    board_tensor, label = dataset[3]
    assert label[2] == 1.0
    assert board_tensor.sum().item() == 64
