"""Cluster IPL batters into playing styles using career-level features."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

MIN_BALLS = 500
K = 5  # number of styles (we'll check this below)

df = pd.read_parquet("data/processed/balls.parquet")
df = df[(df["innings"] <= 2) & df["wides"].isna()].copy()   # balls the batter faced
df["year"] = df["start_date"].dt.year
df["over"] = np.floor(df["ball"]).astype(int)
df["is_four"] = (df["runs_off_bat"] == 4) & df["non_boundary"].isna()
df["is_six"] = df["runs_off_bat"] == 6
df["is_dot"] = df["runs_off_bat"] == 0
df["out"] = (df["player_dismissed"] == df["striker"]) & (df["wicket_type"] != "retired hurt")
df["phase"] = pd.cut(df["over"], [-1, 5, 15, 19], labels=["pp", "mid", "death"])

# League scoring rate per year, so players from different eras are comparable
league_sr = df.groupby("year")["runs_off_bat"].mean()
df["lg_runs"] = df["year"].map(league_sr)

p = df.groupby("striker").agg(
    balls=("runs_off_bat", "size"),
    runs=("runs_off_bat", "sum"),
    exp_runs=("lg_runs", "sum"),
    fours=("is_four", "sum"),
    sixes=("is_six", "sum"),
    dots=("is_dot", "sum"),
    outs=("out", "sum"),
)
p = p[p["balls"] >= MIN_BALLS].copy()

p["rel_sr"] = p["runs"] / p["exp_runs"]                 # strike rate vs league, era-adjusted
p["four_pct"] = p["fours"] / p["balls"] * 100
p["six_pct"] = p["sixes"] / p["balls"] * 100
p["dot_pct"] = p["dots"] / p["balls"] * 100
p["balls_per_out"] = (p["balls"] / p["outs"].replace(0, np.nan)).clip(upper=40).fillna(40)

shares = df.groupby(["striker", "phase"], observed=True).size().unstack(fill_value=0)
shares = shares.div(shares.sum(axis=1), axis=0)
p["pp_share"] = shares["pp"]
p["death_share"] = shares["death"]

FEATS = ["rel_sr", "four_pct", "six_pct", "dot_pct", "balls_per_out", "pp_share", "death_share"]
X = StandardScaler().fit_transform(p[FEATS])

print(f"{len(p)} batters with {MIN_BALLS}+ balls\n")
print("How many styles fit best? (higher silhouette = cleaner groups)")
for k in range(3, 8):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    print(f"  k={k}: silhouette = {silhouette_score(X, labels):.3f}")

p["cluster"] = KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(X)

# 2D map for the dashboard
coords = PCA(n_components=2, random_state=42).fit_transform(X)
p["pc1"], p["pc2"] = coords[:, 0], coords[:, 1]

print(f"\nCluster profiles (K={K}):")
profile = p.groupby("cluster")[FEATS].mean().round(2)
profile["players"] = p.groupby("cluster").size()
print(profile.to_string())

print("\nMost-experienced players in each cluster:")
for c in range(K):
    names = p[p["cluster"] == c].sort_values("balls", ascending=False).head(6).index.tolist()
    print(f"  Cluster {c}: {', '.join(names)}")

p.reset_index().rename(columns={"striker": "player"}).to_parquet(
    "data/processed/batter_clusters.parquet", index=False
)