from typing import Generator, List
from pathlib import Path
import pandas as pd


def puzzle_csv_iterator_pd(
    fname: str | Path,
    batch_rows: int = 256,
) -> Generator[pd.DataFrame]:
    for chunk in pd.read_csv(fname, chunksize=batch_rows):
        yield chunk


def puzzle_csv_iterator(
    fname: str | Path,
) -> Generator[List[str]]:
    with open(fname, "r") as puzzles_f:
        next(puzzles_f)
        for line in puzzles_f:
            yield line.strip().split(",")
