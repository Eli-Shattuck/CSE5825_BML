from beschess.utils import tensor_to_board
import numpy as np

import chess.svg

TAG_NAMES = [
    "Quiet",
    "MatingNet",
    "SpecialMove",
    "Promotion",
    "DoubleAttack",
    "LinearAttack",
    "Punishment",
    "ForcingMove",
]
def get_tag_string(label_vector):
    active_indices = np.where(label_vector > 0)[0]
    return [TAG_NAMES[i] for i in active_indices]

puzzles = [ 81640,  67666,  52157,  63773,  70404, 164997, 127643, 146460, 50792,  75647]

puzzle_dataset = np.load('../data/test_set_embeddings_vit_puzzle_only_boards.npy')
label_dataset = np.load('../data/test_set_embeddings_vit_puzzle_only_labels.npy')

for i, pid in enumerate(puzzles):

    print(i, get_tag_string(label_dataset[pid]))

    board = tensor_to_board(puzzle_dataset[pid])
    svg = chess.svg.board(board)

    with open(f'rec_{i}.svg', 'w') as out_f:
        out_f.write(svg)

    