import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import load_clean, split
from src.models import make_windows, fit_xgb, persistence_predict

FEATS = ["pm25", "temp", "RH", "rain", "wind", "pressure", "wd_sin", "wd_cos"]
LOOKBACK = 24
HORIZON = 1


def main(path="data/archive__1_.zip"):
    df = load_clean(path)
    tr, va, te = split(df)

    X_tr, y_tr = make_windows(tr, FEATS, lookback=LOOKBACK, horizon=HORIZON)
    X_va, y_va = make_windows(va, FEATS, lookback=LOOKBACK, horizon=HORIZON)
    X_te, y_te = make_windows(te, FEATS, lookback=LOOKBACK, horizon=HORIZON)
    print("windows:", X_tr.shape, X_va.shape, X_te.shape)

    m = fit_xgb(X_tr, y_tr, X_va, y_va)
    p = m.predict(X_te)

    print(f"best iteration: {m.best_iteration}")
    print(f"MAE:  {mean_absolute_error(y_te, p):.3f}")
    print(f"RMSE: {mean_squared_error(y_te, p) ** 0.5:.3f}")
    print(f"R2:   {r2_score(y_te, p):.3f}")

    # persistence baseline (same test windows)
    pp, pt = persistence_predict(te, lookback=LOOKBACK, horizon=HORIZON)
    print("--- persistence baseline ---")
    print(f"MAE:  {mean_absolute_error(pt, pp):.3f}")
    print(f"RMSE: {mean_squared_error(pt, pp) ** 0.5:.3f}")
    print(f"R2:   {r2_score(pt, pp):.3f}")


if __name__ == "__main__":
    main()