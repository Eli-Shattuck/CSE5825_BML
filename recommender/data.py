from beschess.load import load_game, load_puzzle
from beschess.analysis import StockFish, StockFishConfig, is_puzzle
import beschess.components.net.vit as vit
from beschess.utils import board_to_packed, packed_to_tensor
from chess import Board
import chess

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm
from itertools import islice
from pathlib import Path

def build_model(checkpoint_path='../checkpoints/best_checkpoint.pth', device=torch.device("cpu")) -> nn.Module:
    return vit.get_interpretable_vit(path_to_weights=checkpoint_path)

def run_model(model, board) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        board_tensor = packed_to_tensor(board_to_packed(board))
        batch = board_tensor.reshape((1,) + board_tensor.shape) # batch with one tensor
        inputs = torch.from_numpy(batch).float()  # CPU tensors
        out = model(inputs)
        if isinstance(out, tuple):
            out = out[0]
        return out.numpy()


def generate_dataset() -> pd.DataFrame:
    games_fname = '../data/lichess_db_standard_rated_2013-01.pgn.zst'
    ngames = 121332
    puzzles_fname = '../data/test_set_embeddings_vit_puzzle_only.npy'

    PUZZLE_RATING_DIFF = 100
    INTERACTION_THRESHOLD = 0.4
    ALPHA = 100

    stockfish = StockFish(StockFishConfig(nodes=5000, threads=16))

    games = load_game.load_game_zstd(games_fname)

    dataset_dict: dict[str, list[int|float]] = { 
        'user':[], 
        'item':[], 
        'label':[],
        'username':[]
    }

    embedding_model = build_model()

    user_puzzles = {}
    take = 100_000
    for game in tqdm(islice(games, take), total=take):
        w_name = game.headers.get('White')
        b_name = game.headers.get('Black')

        board = game.board()

        score = None
        prev_score = None
        prev_board = None

        for move in tqdm(game.mainline_moves(), leave=False):
            prev_score = score
            prev_board = board.copy(stack=False)

            board.push(move)
            if board.is_game_over():
                continue
            
            try:
                is_puz, puz_score, score, _ = is_puzzle(stockfish, board, amax_cp_diff=PUZZLE_RATING_DIFF, return_score=True)
                if not is_puz:
                    continue
            except Exception as e:
                print(f"Engine analysis error: {e}")
                continue

            if prev_board is not None and prev_score is not None:
                b_copy = board.copy(stack=False)
                name = w_name if board.turn == chess.BLACK else b_name # flipped because we are evealuating for previous board
                embedding = run_model(embedding_model, prev_board)

                user_dict = user_puzzles.setdefault(name, {
                    'puzzle_score':[],
                    'pscore':[],
                    'score':[],
                    'pboard':[],
                    'board':[],
                    'embedding':[]
                })
                
                user_dict['puzzle_score'].append(puz_score)
                user_dict['pscore'].append(prev_score)
                user_dict['score'].append(-score)
                user_dict['pboard'].append(prev_board)
                user_dict['board'].append(b_copy)
                user_dict['embedding'].append(embedding)

    puzzles = np.load(puzzles_fname)
    # print(f'npuzzles: {puzzles.shape}')

    uid_counter = 0
    uids = {}

    for user, user_dict in tqdm(user_puzzles.items(), leave=False):
        if user not in uids:
            uids[user] = uid_counter
            uid_counter += 1
        uid = uids[user]

        embeddings = np.vstack(user_dict['embedding'])
        delta_s = np.tanh( (np.array(user_dict['score']) - np.array(user_dict['pscore'])) / ALPHA )

        phi = cosine_similarity(embeddings, puzzles)
        I_u:np.ndarray = - (delta_s[:, None] * phi).sum(axis=0)
        I_u /= len(delta_s)
        
        # print(I_u.min(), I_u.max(), I_u.mean())

        pids = np.where(np.abs(I_u) > INTERACTION_THRESHOLD)[0]
        for pid in pids:
            dataset_dict['user'].append(uid)
            dataset_dict['item'].append(pid)
            dataset_dict['label'].append(I_u[pid])
            dataset_dict['username'].append(user)

    stockfish.quit()

    dataset = pd.DataFrame(dataset_dict)
    return dataset

def load_dataset(fname:str='../data/interactions.csv') -> pd.DataFrame:
    interaction_matrix_path = Path(fname)

    if interaction_matrix_path.exists():
        dataset = pd.read_csv(interaction_matrix_path)
    else:
        dataset = generate_dataset()
        dataset.to_csv(interaction_matrix_path)
    
    return dataset

if __name__ == '__main__':
    dataset = load_dataset()
    print(dataset.head())