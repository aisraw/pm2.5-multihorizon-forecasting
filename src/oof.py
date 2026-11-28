import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data import load_clean
from src.models import make_windows, fit_xgb, build_lstm_attn, fit_net

FEATS = ["pm25", "temp", "RH", "rain", "wind", "pressure", "wd_sin", "wd_cos"]
LOOKBACK = 24
SEED = 42

FOLDS = [
    ("2023-06-30 23:59", "2023-07-01", "2023-12-31 23:59"),
    ("2023-12-31 23:59", "2024-01-01", "2024-06-30 23:59"),
    ("2024-06-30 23:59", "2024-07-01", "2024-12-31 23:59"),
]


def sub(df, start=None, end=None):
    m = pd.Series(True, index=df.index)
    if start is not None:
        m &= df["timestamp"] >= pd.Timestamp(start)
    if end is not None:
        m &= df["timestamp"] <= pd.Timestamp(end)
    return df[m]


def main(horizon, path="data/archive__1_.zip"):
    df = load_clean(path)
    rows = []

    for train_end, pred_start, pred_end in FOLDS:
        tr = sub(df, end=train_end)
        pr = sub(df, start=pred_start, end=pred_end)

        cut = int(len(tr) * 0.9)
        tr_fit, tr_val = tr.iloc[:cut], tr.iloc[cut:]

        # XGBoost (2D)
        Xa, ya = make_windows(tr_fit, FEATS, lookback=LOOKBACK, horizon=horizon, flat=True)
        Xv, yv = make_windows(tr_val, FEATS, lookback=LOOKBACK, horizon=horizon, flat=True)
        Xp, yp = make_windows(pr, FEATS, lookback=LOOKBACK, horizon=horizon, flat=True)
        xgb = fit_xgb(Xa, ya, Xv, yv, seed=SEED)
        p_xgb = xgb.predict(Xp)

        # LSTM+Attn (3D)
        La, lya = make_windows(tr_fit, FEATS, lookback=LOOKBACK, horizon=horizon, flat=False)
        Lv, lyv = make_windows(tr_val, FEATS, lookback=LOOKBACK, horizon=horizon, flat=False)
        Lp, lyp = make_windows(pr, FEATS, lookback=LOOKBACK, horizon=horizon, flat=False)
        tf.keras.backend.clear_session()
        net = fit_net(build_lstm_attn(LOOKBACK, len(FEATS)), La, lya, Lv, lyv, seed=SEED)
        p_lstm = net.predict(Lp, verbose=0).ravel()

        assert np.allclose(yp, lyp), "target mismatch between XGB and LSTM windows"
        for i in range(len(yp)):
            rows.append({"fold_end": train_end, "y": yp[i],
                         "p_xgb": p_xgb[i], "p_lstm": p_lstm[i]})
        print(f"h={horizon} done fold ending {train_end}: {len(yp)} rows", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(f"results/oof_{horizon}h.csv", index=False)
    print(out.describe().round(3).to_string())


if __name__ == "__main__":
    h = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(h)