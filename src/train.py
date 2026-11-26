import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import load_clean, split
from src.models import make_windows, fit_xgb, persistence_predict

FEATS = ["pm25", "temp", "RH", "rain", "wind", "pressure", "wd_sin", "wd_cos"]
LOOKBACK = 24
HORIZONS = [1, 3, 6]


def score(y, p):
    return (mean_absolute_error(y, p),
            mean_squared_error(y, p) ** 0.5,
            r2_score(y, p))


def main(path="data/archive__1_.zip"):
    df = load_clean(path)
    tr, va, te = split(df)

    rows = []
    for h in HORIZONS:
        X_tr, y_tr = make_windows(tr, FEATS, lookback=LOOKBACK, horizon=h)
        X_va, y_va = make_windows(va, FEATS, lookback=LOOKBACK, horizon=h)
        X_te, y_te = make_windows(te, FEATS, lookback=LOOKBACK, horizon=h)

        m = fit_xgb(X_tr, y_tr, X_va, y_va)
        p = m.predict(X_te)
        pp, pt = persistence_predict(te, lookback=LOOKBACK, horizon=h)

        mae, rmse, r2 = score(y_te, p)
        pmae, prmse, pr2 = score(pt, pp)
        rows.append((h, "XGBoost", mae, rmse, r2, m.best_iteration))
        rows.append((h, "Persistence", pmae, prmse, pr2, "-"))

    print(f"{'h':>3} {'model':<12} {'MAE':>7} {'RMSE':>7} {'R2':>7} {'best_it':>8}")
    for h, name, mae, rmse, r2, bi in rows:
        print(f"{h:>3} {name:<12} {mae:>7.3f} {rmse:>7.3f} {r2:>7.3f} {str(bi):>8}")


if __name__ == "__main__":
    main()