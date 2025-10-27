#!/usr/bin/env python

# SBATCH --partition=priority
# SBATCH --ntasks=1
# SBATCH --cpus-per-task=8
# SBATCH --output=.cache/analysis_out_%j.log
# SBATCH --error=.cache/analysis_err_%j.log
# SBATCH --time=0

from beschess.load import load_game
from beschess.analysis import engine_analysis
import pandas as pd

num_games = load_game.get_n_games_pgn(r"../data/lichess_db_standard_rated_2013-01.pgn")

analyzers = map(
    lambda g: engine_analysis(g, depth=12),
    load_game.load_games_pgn(r"../data/lichess_db_standard_rated_2013-01.pgn"),
)

user_dict = {}
for i, analyzer in enumerate(analyzers):
    (white, black), _, _, score = next(analyzer)
    user_dict.setdefault(white, []).append([score])
    user_dict.setdefault(black, []).append([-score])

    for (white, black), _, move, score in analyzer:
        user_dict[white][-1].append(score)
        user_dict[black][-1].append(-score)

    print(f"Processed game {i + 1} of {num_games}")


# Save to CSV
rows = []
for user, games in user_dict.items():
    for game in games:
        for move_number, score in enumerate(game):
            rows.append({"user": user, "move_number": move_number + 1, "score": score})


df = pd.DataFrame(rows)
df.to_csv("user_game_scores.csv", index=False)
