"""Load and clean IPL ball-by-ball data from Cricsheet."""
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/ipl")
OUT_FILE = Path("data/processed/balls.parquet")

# Files in the Cricsheet zip that are NOT individual match files
SKIP_FILES = {"all_matches.csv"}


def load_ball_by_ball(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    files = [
        f for f in raw_dir.glob("*.csv")
        if not f.name.endswith("_info.csv") and f.name not in SKIP_FILES
    ]
    # Reading season as text avoids mixed-type problems (e.g. 2009 vs "2009/10")
    frames = [pd.read_csv(f, dtype={"season": str}, low_memory=False) for f in files]
    balls = pd.concat(frames, ignore_index=True)
    balls["start_date"] = pd.to_datetime(balls["start_date"])
    return balls


def check_data(balls: pd.DataFrame) -> None:
    """Fail loudly if the data looks wrong."""
    n_matches = balls["match_id"].nunique()
    per_match = len(balls) / n_matches
    assert balls.duplicated().sum() == 0, "Duplicate rows found"
    assert 200 < per_match < 300, f"Unexpected balls per match: {per_match:.1f}"
    print(f"OK: {n_matches} matches, {len(balls)} deliveries, {per_match:.1f} balls/match")


if __name__ == "__main__":
    balls = load_ball_by_ball()
    check_data(balls)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    balls.to_parquet(OUT_FILE, index=False)
    print("Saved to", OUT_FILE)