from beschess.load import load_game, load_puzzle
from beschess.analysis import StockFish, StockFishConfig, is_puzzle
import beschess.components.net.vit as vit
from beschess.utils import board_to_packed, packed_to_tensor
from chess import Board
import chess

from data import build_model, run_model

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm
from itertools import islice
from pathlib import Path

def save_user_puzzles(username:str='german11', path:str='../data/german11_user_puzzles.npy') -> None:
    games = load_game.load_games_pgn('../data/games.pgn')

    w_name = lambda g: g.headers.get('White')
    b_name = lambda g: g.headers.get('Black')

    user_games = ( game for game in games if username in [w_name(game), b_name(game)] )

    stockfish = StockFish(StockFishConfig(nodes=5000, threads=16))

    PUZZLE_RATING_DIFF = 100

    embeddings = []
    embedding_model = build_model()

    for game in tqdm(user_games, total=1611):
        user_color = chess.WHITE if username == w_name(game) else chess.BLACK

        board = game.board()
        for move in tqdm(game.mainline_moves(),leave=False):
            board.push(move)
            if board.is_game_over():
                continue
            
            if board.turn != user_color: continue

            try:
                if not is_puzzle(stockfish, board, amax_cp_diff=PUZZLE_RATING_DIFF):
                    continue
            except Exception as e:
                print(f"Engine analysis error: {e}")
                continue

            embedding = run_model(embedding_model, board)
            embeddings.append(embedding)
    
    stockfish.quit()
    user_puzzles = np.vstack(embeddings)
    np.save(path, user_puzzles)

if __name__ == '__main__':
    save_user_puzzles()