import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import load_clean, split
from src.models import (make_windows, fit_xgb, persistence_predict,
                        fit_lstm)

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
        # XGBoost (2D flat windows)
        X_tr, y_tr = make_windows(tr, FEATS, lookback=LOOKBACK, horizon=h, flat=True)
        X_va, y_va = make_windows(va, FEATS, lookback=LOOKBACK, horizon=h, flat=True)
        X_te, y_te = make_windows(te, FEATS, lookback=LOOKBACK, horizon=h, flat=True)
        m = fit_xgb(X_tr, y_tr, X_va, y_va)
        p_xgb = m.predict(X_te)

        # LSTM (3D sequence windows)
        L_tr, ly_tr = make_windows(tr, FEATS, lookback=LOOKBACK, horizon=h, flat=False)
        L_va, ly_va = make_windows(va, FEATS, lookback=LOOKBACK, horizon=h, flat=False)
        L_te, ly_te = make_windows(te, FEATS, lookback=LOOKBACK, horizon=h, flat=False)
        net = fit_lstm(L_tr, ly_tr, L_va, ly_va)
        p_lstm = net.predict(L_te, verbose=0).ravel()

        # Persistence
        pp, pt = persistence_predict(te, lookback=LOOKBACK, horizon=h)

        rows.append((h, "XGBoost", *score(y_te, p_xgb)))
        rows.append((h, "LSTM", *score(ly_te, p_lstm)))
        rows.append((h, "Persistence", *score(pt, pp)))

    print(f"{'h':>3} {'model':<12} {'MAE':>7} {'RMSE':>7} {'R2':>7}")
    for h, name, mae, rmse, r2 in rows:
        print(f"{h:>3} {name:<12} {mae:>7.3f} {rmse:>7.3f} {r2:>7.3f}")


if __name__ == "__main__":
    main()