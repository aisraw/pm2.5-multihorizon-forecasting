import os
import sys
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import load_clean, split
from src.models import (make_windows, fit_xgb, persistence_predict,
                        build_lstm, build_lstm_attn, build_lstm_transformer,
                        fit_net)

FEATS = ["pm25", "temp", "RH", "rain", "wind", "pressure", "wd_sin", "wd_cos"]
LOOKBACK = 24
HORIZONS = [1, 3, 6]
MODELS = ["XGBoost", "LSTM", "LSTM+Attn", "LSTM+Transformer", "Persistence"]
DEEP = {
    "LSTM": build_lstm,
    "LSTM+Attn": build_lstm_attn,
    "LSTM+Transformer": build_lstm_transformer,
}
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
OUT = f"results/seed{SEED}.csv"


def score(y, p):
    return (mean_absolute_error(y, p),
            mean_squared_error(y, p) ** 0.5,
            r2_score(y, p))


def load_done():
    if os.path.exists(OUT):
        d = pd.read_csv(OUT)
        return set(zip(d["h"].astype(int), d["model"]))
    return set()


def save_row(row):
    new = not os.path.exists(OUT)
    pd.DataFrame([row]).to_csv(OUT, mode="a", header=new, index=False)


def run_one(h, name, tr, va, te, seed):
    if name == "XGBoost":
        X_tr, y_tr = make_windows(tr, FEATS, lookback=LOOKBACK, horizon=h, flat=True)
        X_va, y_va = make_windows(va, FEATS, lookback=LOOKBACK, horizon=h, flat=True)
        X_te, y_te = make_windows(te, FEATS, lookback=LOOKBACK, horizon=h, flat=True)
        m = fit_xgb(X_tr, y_tr, X_va, y_va, seed=seed)
        return score(y_te, m.predict(X_te))

    if name == "Persistence":
        pp, pt = persistence_predict(te, lookback=LOOKBACK, horizon=h)
        return score(pt, pp)

    L_tr, ly_tr = make_windows(tr, FEATS, lookback=LOOKBACK, horizon=h, flat=False)
    L_va, ly_va = make_windows(va, FEATS, lookback=LOOKBACK, horizon=h, flat=False)
    L_te, ly_te = make_windows(te, FEATS, lookback=LOOKBACK, horizon=h, flat=False)
    tf.keras.backend.clear_session()
    net = fit_net(DEEP[name](LOOKBACK, len(FEATS)),
                  L_tr, ly_tr, L_va, ly_va, seed=seed)
    return score(ly_te, net.predict(L_te, verbose=0).ravel())


def main(path="data/archive__1_.zip"):
    os.makedirs("results", exist_ok=True)
    df = load_clean(path)
    tr, va, te = split(df)
    done = load_done()

    for h in HORIZONS:
        for name in MODELS:
            if (h, name) in done:
                print(f"skip: seed={SEED} h={h} {name}", flush=True)
                continue
            mae, rmse, r2 = run_one(h, name, tr, va, te, SEED)
            save_row({"h": h, "model": name, "MAE": mae, "RMSE": rmse,
                      "R2": r2, "seed": SEED})
            print(f"done: seed={SEED} h={h} {name} MAE={mae:.3f}", flush=True)

    print(pd.read_csv(OUT).to_string(index=False))


if __name__ == "__main__":
    main()