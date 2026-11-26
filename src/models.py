import numpy as np
from xgboost import XGBRegressor


def make_windows(df, feats, target="pm25", lookback=24, horizon=1):
    """Station-wise sliding window: last `lookback` hours -> pm25 at t+horizon."""
    X, y = [], []
    for _, g in df.groupby("station"):
        g = g.sort_values("timestamp")
        F = g[feats].values
        t = g[target].values
        for i in range(lookback, len(g) - horizon + 1):
            X.append(F[i - lookback:i].ravel())
            y.append(t[i - 1 + horizon])
    return np.array(X), np.array(y)


def fit_xgb(X_tr, y_tr, X_va, y_va, seed=42):
    m = XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        early_stopping_rounds=30, random_state=seed,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m