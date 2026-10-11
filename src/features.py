"""Turn ball-by-ball data into one row per batter per season."""
from pathlib import Path

import numpy as np
import pandas as pd

IN_FILE = Path("data/processed/balls.parquet")
OUT_FILE = Path("data/processed/batter_seasons.parquet")


def build_batter_seasons(balls: pd.DataFrame) -> pd.DataFrame:
    df = balls.copy()

    # IPL seasons sit inside one calendar year, so use that as a clean season key
    # (this also fixes labels like "2007/08" and "2009/10")
    df["year"] = df["start_date"].dt.year

    # A ball counts as "faced" by the batter unless it was a wide
    df["legal_ball"] = df["wides"].isna()

    df["is_four"] = (df["runs_off_bat"] == 4) & df["non_boundary"].isna()
    df["is_six"] = df["runs_off_bat"] == 6
    df["is_dot"] = df["legal_ball"] & (df["runs_off_bat"] == 0)

    # The striker was dismissed (ignore run-outs of the non-striker and retired hurt)
    df["striker_out"] = (
        (df["player_dismissed"] == df["striker"])
        & (df["wicket_type"] != "retired hurt")
    )

    # Which over was it? ball is like 0.1, 0.2 ... 19.6, so floor() gives the over (0-19)
    df["over"] = np.floor(df["ball"]).astype(int)

    grouped = df.groupby(["striker", "year"])
    out = grouped.agg(
        matches=("match_id", "nunique"),
        runs=("runs_off_bat", "sum"),
        balls_faced=("legal_ball", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
        dots=("is_dot", "sum"),
        dismissals=("striker_out", "sum"),
    ).reset_index()

    # Derived features (the numbers the models will actually use)
    out["strike_rate"] = out["runs"] / out["balls_faced"] * 100
    out["average"] = out["runs"] / out["dismissals"].replace(0, np.nan)
    out["boundary_pct"] = (out["fours"] + out["sixes"]) / out["balls_faced"] * 100
    out["dot_pct"] = out["dots"] / out["balls_faced"] * 100
    out["balls_per_dismissal"] = out["balls_faced"] / out["dismissals"].replace(0, np.nan)

    return out.rename(columns={"striker": "player"})


if __name__ == "__main__":
    balls = pd.read_parquet(IN_FILE)
    features = build_batter_seasons(balls)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUT_FILE, index=False)
    print(features.shape)
    print(features.sort_values("runs", ascending=False).head(5).to_string())