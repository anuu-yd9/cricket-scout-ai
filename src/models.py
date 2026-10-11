"""Train and compare models for next-season relative strike rate."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import FEATURES, TARGET, build_dataset, split_by_time

seasons = pd.read_parquet("data/processed/batter_seasons.parquet")
data = build_dataset(seasons)
train, test = split_by_time(data)

# Some features can be missing (e.g. career average); fill with league-average (1.0)
X_train = train[FEATURES].fillna(1.0)
X_test = test[FEATURES].fillna(1.0)
y_train, y_test = train[TARGET], test[TARGET]

results = []

# ---- Baselines (no learning) ----
results.append(("Baseline: same as last season", mean_absolute_error(y_test, test["prev_rel_sr"])))
results.append(("Baseline: career average", mean_absolute_error(y_test, test["career_rel_sr"].fillna(1.0))))
results.append(("Baseline: league average (1.0)", mean_absolute_error(y_test, np.ones(len(y_test)))))

# ---- Learned models ----
models = {
    "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge regression": make_pipeline(StandardScaler(), Ridge(alpha=10)),
    "Random forest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=10, random_state=42, n_jobs=-1
    ),
    "Gradient boosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42
    ),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results.append((name, mean_absolute_error(y_test, pred)))

table = pd.DataFrame(results, columns=["model", "test_MAE"]).sort_values("test_MAE")
print(table.to_string(index=False))

best_baseline = table[table["model"].str.startswith("Baseline")]["test_MAE"].min()
print(f"\nBest baseline MAE: {best_baseline:.3f}")

# Which features does the random forest rely on?
rf = models["Random forest"]
importance = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nRandom forest feature importance:")
print(importance.round(3).to_string())