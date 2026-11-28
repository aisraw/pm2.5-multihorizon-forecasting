import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import load_clean, split
from src.models import (make_windows, fit_xgb, build_lstm_attn, fit_net,
                        persistence_predict)

FEATS = ["pm25", "temp", "RH", "rain", "wind", "pressure", "wd_sin", "wd_cos"]
LOOKBACK = 24
SEED = 42
HORIZONS = [1, 3, 6]
OUT = "results/hybrid_results.csv"


def score(y, p):
    return (mean_absolute_error(y, p),
            mean_squared_error(y, p) ** 0.5,
            r2_score(y, p))


def run_horizon(h, tr, va, te):
    Xtr, ytr = make_windows(tr, FEATS, lookback=LOOKBACK, horizon=h, flat=True)
    Xva, yva = make_windows(va, FEATS, lookback=LOOKBACK, horizon=h, flat=True)
    Xte, yte = make_windows(te, FEATS, lookback=LOOKBACK, horizon=h, flat=True)
    xgb = fit_xgb(Xtr, ytr, Xva, yva, seed=SEED)
    p_xgb = xgb.predict(Xte)

    Ltr, lytr = make_windows(tr, FEATS, lookback=LOOKBACK, horizon=h, flat=False)
    Lva, lyva = make_windows(va, FEATS, lookback=LOOKBACK, horizon=h, flat=False)
    Lte, lyte = make_windows(te, FEATS, lookback=LOOKBACK, horizon=h, flat=False)
    tf.keras.backend.clear_session()
    net = fit_net(build_lstm_attn(LOOKBACK, len(FEATS)),
                  Ltr, lytr, Lva, lyva, seed=SEED)
    p_lstm = net.predict(Lte, verbose=0).ravel()

    pp, pt = persistence_predict(te, LOOKBACK, h)
    assert np.allclose(yte, pt) and np.allclose(yte, lyte), "test alignment mismatch"

    oof = pd.read_csv(f"results/oof_{h}h.csv")
    y_oof = oof["y"].values
    ws = np.linspace(0, 1, 101)
    maes = [mean_absolute_error(y_oof, w * oof["p_xgb"] + (1 - w) * oof["p_lstm"])
            for w in ws]
    w_best = ws[int(np.argmin(maes))]
    p_blend = w_best * p_xgb + (1 - w_best) * p_lstm

    meta = Ridge(alpha=1.0).fit(oof[["p_xgb", "p_lstm"]].values, y_oof)
    p_stack = meta.predict(np.column_stack([p_xgb, p_lstm]))

    rows = []
    for name, p, y in [("XGBoost", p_xgb, yte),
                       ("LSTM+Attn", p_lstm, yte),
                       (f"Blend (w_xgb={w_best:.2f})", p_blend, yte),
                       ("Ridge stack", p_stack, yte),
                       ("Persistence", pp, pt)]:
        mae, rmse, r2 = score(y, p)
        rows.append({"h": h, "model": name, "MAE": mae, "RMSE": rmse, "R2": r2,
                     "ridge_coef_xgb": meta.coef_[0] if name == "Ridge stack" else None,
                     "ridge_coef_lstm": meta.coef_[1] if name == "Ridge stack" else None})
    return rows


def main(path="data/archive__1_.zip"):
    df = load_clean(path)
    tr, va, te = split(df)

    all_rows = []
    for h in HORIZONS:
        print(f"--- horizon {h}h ---", flush=True)
        all_rows += run_horizon(h, tr, va, te)

    out = pd.DataFrame(all_rows)
    out.to_csv(OUT, index=False)
    print(out[["h", "model", "MAE", "RMSE", "R2"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()