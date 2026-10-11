"""Build the supervised learning dataset: history -> next-season relative strike rate."""
from pathlib import Path

import numpy as np
import pandas as pd

IN_FILE = Path("data/processed/batter_seasons.parquet")
OUT_FILE = Path("data/processed/model_dataset.parquet")

MIN_BALLS = 60          # ignore tiny samples
TRAIN_END_YEAR = 2023   # train on <= 2023, test on later seasons

FEATURES = [
    "prev_rel_sr", "prev_boundary_pct", "prev_dot_pct",
    "prev_balls_faced", "prev_matches",
    "career_rel_sr", "seasons_before", "years_since_prev",
]
TARGET = "target_rel_sr"


def build_dataset(seasons: pd.DataFrame) -> pd.DataFrame:
    df = seasons.copy()

    # League-average strike rate for every year (removes the "era effect")
    league = df.groupby("year").agg(lr=("runs", "sum"), lb=("balls_faced", "sum"))
    league["league_sr"] = league["lr"] / league["lb"] * 100
    df = df.merge(league[["league_sr"]], on="year")
    df["rel_sr"] = df["strike_rate"] / df["league_sr"]

    df = df[df["balls_faced"] >= MIN_BALLS].sort_values(["player", "year"])
    df = df.reset_index(drop=True)

    # Ball-weighted running totals, used for the career average BEFORE each season
    df["_w"] = df["rel_sr"] * df["balls_faced"]
    g = df.groupby("player")

    for col in ["rel_sr", "boundary_pct", "dot_pct", "balls_faced", "matches"]:
        df[f"prev_{col}"] = g[col].shift(1)          # previous qualified season
    df["years_since_prev"] = df["year"] - g["year"].shift(1)
    df["seasons_before"] = g.cumcount()               # seasons played before this one

    cum_w = g["_w"].cumsum() - df["_w"]               # subtract the current season
    cum_b = g["balls_faced"].cumsum() - df["balls_faced"]
    df["career_rel_sr"] = cum_w / cum_b.replace(0, np.nan)

    df = df.rename(columns={"rel_sr": TARGET})
    df = df.dropna(subset=["prev_rel_sr"])            # first-ever seasons have no history

    return df[["player", "year", "league_sr", *FEATURES, TARGET]].reset_index(drop=True)


def split_by_time(df: pd.DataFrame):
    train = df[df["year"] <= TRAIN_END_YEAR]
    test = df[df["year"] > TRAIN_END_YEAR]
    return train, test


if __name__ == "__main__":
    seasons = pd.read_parquet(IN_FILE)
    data = build_dataset(seasons)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUT_FILE, index=False)

    train, test = split_by_time(data)
    print("total rows:", len(data), "| train:", len(train), "| test:", len(test))

    # The simplest possible "model": predict next season = last season.
    for name, part in [("train", train), ("test", test)]:
        mae = (part[TARGET] - part["prev_rel_sr"]).abs().mean()
        print(f"Baseline MAE on {name}: {mae:.3f}")
    print(data.head().to_string())