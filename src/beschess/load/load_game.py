from chess import pgn
from typing import Generator, List
from pathlib import Path
import pandas as pd


def load_games_pgn(fname: str | Path) -> Generator[pgn.Game]:
    with open(fname, "r") as games_f:
        while game := pgn.read_game(games_f):
            yield game


def load_games_csv(fname: str | Path) -> pd.DataFrame:
    raise NotImplementedError("not implemented")


def load_headers_pgn(fname: str | Path) -> Generator[pgn.Headers]:
    with open(fname, "r") as games_f:
        while head := pgn.read_headers(games_f):
            yield head


def load_headers_csv(fname: str | Path) -> pd.DataFrame:
    return pd.read_csv(fname)


def to_csv(headers: Generator[pgn.Headers], output_path: str | Path) -> None:
    data = {
        "Game Type": [],
        "White.Name": [],
        "Black.Name": [],
        "Winner": [],
        "White.Elo": [],
        "Black.Elo": [],
        "Time Control": [],
    }

    for head in headers:
        data["Game Type"].append(head.get("Event"))
        data["White.Name"].append(head.get("White"))
        data["Black.Name"].append(head.get("Black"))
        data["Winner"].append(head.get("Result"))
        we = head.get("WhiteElo")
        data["White.Elo"].append(None if we == "?" else we)

        be = head.get("BlackElo")
        data["Black.Elo"].append(None if be == "?" else be)
        # data["Black.Elo"].append(head.get("BlackElo"))
        data["Time Control"].append(head.get("TimeControl"))

    df = pd.DataFrame(data)
    df.to_csv(output_path)


if __name__ == "__main__":
    games = load_games_pgn(r"../../../data/lichess_db_standard_rated_2013-01.pgn")
    to_csv(
        load_headers_pgn(r"../../../data/lichess_db_standard_rated_2013-01.pgn"),
        r"../../../data/headers.csv",
    )
    # head = load_headers_csv(r"../../../data/headers.csv")
    # print(head)
