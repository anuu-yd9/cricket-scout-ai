"""Walk-forward evaluation: train on the past, test on each following year."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from dataset import FEATURES, TARGET, build_dataset

data = build_dataset(pd.read_parquet("data/processed/batter_seasons.parquet"))

rows = []
for test_year in range(2014, 2027):
    train = data[data["year"] < test_year]
    test = data[data["year"] == test_year]
    if len(test) < 15 or len(train) < 100:
        continue

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42
    )
    model.fit(train[FEATURES].fillna(1.0), train[TARGET])
    pred = model.predict(test[FEATURES].fillna(1.0))
    y = test[TARGET]

    rows.append({
        "test_year": test_year,
        "n_test": len(test),
        "model": mean_absolute_error(y, pred),
        "league_avg": mean_absolute_error(y, np.ones(len(y))),
        "shrunk_career": mean_absolute_error(y, test["shrunk_rel_sr"]),
        "last_season": mean_absolute_error(y, test["prev_rel_sr"]),
    })

res = pd.DataFrame(rows)
print(res.round(3).to_string(index=False))

print("\nAverage MAE across years:")
print(res[["model", "league_avg", "shrunk_career", "last_season"]].mean().round(4).to_string())

wins = (res["model"] < res[["league_avg", "shrunk_career"]].min(axis=1)).sum()
print(f"\nModel beats the best simple baseline in {wins} of {len(res)} years")