from chess import pgn
from typing import Generator, List
from pathlib import Path
import pandas as pd

def load_games(fname: str|Path) -> Generator[pgn.Game]:
    with open(fname, 'r') as games_f:
        while game := pgn.read_game(games_f):
            yield game

def load_headers(fname: str|Path) -> pgn.Headers:
    with open(fname, 'r') as games_f:
        return pgn.read_headers(games_f)

def to_csv(headers: pgn.Headers, output_path: str|Path) -> None:
    csv = { 
        'Game Type':[], 
        'White.Name':[],  
        'Black.Name':[],  
        'Winner': [],
        'White.Elo': [],        
        'Black.Elo': [],
        "TimeControl": []
    }

    print(headers)

if __name__ == '__main__':
    #games = load_games(r'./data/games.pgn')
    to_csv(load_headers(r'./data/games.pgn'), )