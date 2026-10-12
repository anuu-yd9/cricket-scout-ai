"""Expected-runs model: how many runs would a typical batter score in this situation?"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

IN_FILE = Path("data/processed/balls.parquet")
OUT_FILE = Path("data/processed/batter_rae.parquet")
SITUATION = ["innings", "over", "wkts_before", "runs_before",
             "balls_remaining", "runs_needed", "required_rr", "year"]

# NOTE: rows are kept in their original file order (each file lists deliveries in
# the order they were bowled), so we do NOT re-sort here.
df = pd.read_parquet(IN_FILE)
df = df[df["innings"] <= 2].copy()          # drop super overs
df["year"] = df["start_date"].dt.year

df["total_runs"] = df["runs_off_bat"] + df["extras"]
df["is_wicket"] = df["player_dismissed"].notna().astype(int)
df["legal"] = (df["wides"].isna() & df["noballs"].isna()).astype(int)

g = df.groupby(["match_id", "innings"])
df["runs_before"] = g["total_runs"].cumsum() - df["total_runs"]
df["wkts_before"] = g["is_wicket"].cumsum() - df["is_wicket"]
df["balls_before"] = g["legal"].cumsum() - df["legal"]
df["balls_remaining"] = (120 - df["balls_before"]).clip(lower=0)
df["over"] = np.floor(df["ball"]).astype(int)

# Chase information (only exists in the 2nd innings)
first_total = (df[df["innings"] == 1].groupby("match_id")["total_runs"].sum()
               .rename("first_total"))
df = df.merge(first_total, on="match_id", how="left")
chase = df["innings"] == 2
df["runs_needed"] = np.where(chase, df["first_total"] + 1 - df["runs_before"], np.nan)
df["required_rr"] = np.where(
    chase, df["runs_needed"] / df["balls_remaining"].clip(lower=1) * 6, np.nan
)

# Only deliveries the batter actually faced (everything except wides)
faced = df[df["wides"].isna()].copy()
y = faced["runs_off_bat"]

# ---- Honest check: train on old seasons, test on newer ones ----
train = faced["year"] <= 2023
model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=4, random_state=42)
model.fit(faced.loc[train, SITUATION], y[train])
pred = model.predict(faced.loc[~train, SITUATION])
mse_model = mean_squared_error(y[~train], pred)
mse_const = mean_squared_error(y[~train], np.full((~train).sum(), y[train].mean()))
print(f"Test MSE  model: {mse_model:.4f} | constant-average baseline: {mse_const:.4f}")
print(f"Improvement: {(1 - mse_model / mse_const) * 100:.1f}%")

# ---- Final model on all data (it knows nothing about WHO is batting) ----
model.fit(faced[SITUATION], y)
faced["exp_runs"] = model.predict(faced[SITUATION])

# ---- Runs above expected per batter-season ----
bs = (faced.groupby(["striker", "year"])
      .agg(balls=("runs_off_bat", "size"),
           runs=("runs_off_bat", "sum"),
           exp_runs=("exp_runs", "sum"))
      .reset_index().rename(columns={"striker": "player"}))
bs["rae_per100"] = (bs["runs"] - bs["exp_runs"]) / bs["balls"] * 100

league_sr = bs.groupby("year").apply(lambda d: d["runs"].sum() / d["balls"].sum() * 100)
bs["sr_vs_league"] = bs["runs"] / bs["balls"] * 100 - bs["year"].map(league_sr)

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
bs.to_parquet(OUT_FILE, index=False)

q = bs[bs["balls"] >= 300].sort_values("rae_per100", ascending=False)
print("\nTop 10 runs-above-expected seasons (300+ balls):")
print(q.head(10).round(1).to_string(index=False))

# ---- The key test: does this metric carry over to NEXT season better than plain SR? ----
nxt = bs.copy()
nxt["year"] = nxt["year"] - 1
m = bs.merge(nxt, on=["player", "year"], suffixes=("", "_next"))
m = m[(m["balls"] >= 150) & (m["balls_next"] >= 150)]
print(f"\nPlayer-season pairs compared: {len(m)}")
print("Next-season correlation, runs above expected:", round(m["rae_per100"].corr(m["rae_per100_next"]), 3))
print("Next-season correlation, strike rate vs league:", round(m["sr_vs_league"].corr(m["sr_vs_league_next"]), 3))