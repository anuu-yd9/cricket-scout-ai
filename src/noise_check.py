"""Does model accuracy depend on how many balls the target season had?"""
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from dataset import FEATURES, TARGET, build_dataset

data = build_dataset(pd.read_parquet("data/processed/batter_seasons.parquet"))

parts = []
for test_year in range(2014, 2027):
    train = data[data["year"] < test_year]
    test = data[data["year"] == test_year]
    if len(test) < 15 or len(train) < 100:
        continue
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42
    )
    model.fit(train[FEATURES].fillna(1.0), train[TARGET])
    out = test[["player", "year", "balls_faced", TARGET]].copy()
    out["pred_model"] = model.predict(test[FEATURES].fillna(1.0))
    parts.append(out)

p = pd.concat(parts)
p["err_model"] = (p[TARGET] - p["pred_model"]).abs()
p["err_league_avg"] = (p[TARGET] - 1.0).abs()
p["balls_bucket"] = pd.cut(p["balls_faced"], [0, 100, 200, 350, 2000],
                           labels=["60-100", "100-200", "200-350", "350+"])

summary = p.groupby("balls_bucket", observed=True).agg(
    rows=("err_model", "size"),
    model_MAE=("err_model", "mean"),
    league_avg_MAE=("err_league_avg", "mean"),
)
print(summary.round(3).to_string())