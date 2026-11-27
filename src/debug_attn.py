import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score

from src.data import load_clean, split
from src.models import make_windows, build_lstm_attn, fit_net

FEATS = ["pm25", "temp", "RH", "rain", "wind", "pressure", "wd_sin", "wd_cos"]
LOOKBACK, H = 24, 1


def main():
    df = load_clean("data/archive__1_.zip")
    tr, va, te = split(df)
    Xtr, ytr = make_windows(tr, FEATS, lookback=LOOKBACK, horizon=H, flat=False)
    Xva, yva = make_windows(va, FEATS, lookback=LOOKBACK, horizon=H, flat=False)
    Xte, yte = make_windows(te, FEATS, lookback=LOOKBACK, horizon=H, flat=False)

    tf.keras.backend.clear_session()
    net = fit_net(build_lstm_attn(LOOKBACK, len(FEATS)), Xtr, ytr, Xva, yva)
    p = net.predict(Xte, verbose=0).ravel()
    print(f"LSTM+Attn (fixed) 1h  MAE={mean_absolute_error(yte, p):.3f}  "
          f"R2={r2_score(yte, p):.3f}")


if __name__ == "__main__":
    main()