import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBRegressor


def make_windows(df, feats, target="pm25", lookback=24, horizon=1, flat=True):
    """Station-wise sliding window. flat=True -> XGBoost (2D), False -> LSTM (3D)."""
    X, y = [], []
    for _, g in df.groupby("station"):
        g = g.sort_values("timestamp")
        F = g[feats].values
        t = g[target].values
        for i in range(lookback, len(g) - horizon + 1):
            w = F[i - lookback:i]
            X.append(w.ravel() if flat else w)
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


def persistence_predict(df, lookback=24, horizon=1):
    """Naive baseline: prediction = last observed pm25 (t)."""
    preds, truth = [], []
    for _, g in df.groupby("station"):
        g = g.sort_values("timestamp")
        t = g["pm25"].values
        for i in range(lookback, len(g) - horizon + 1):
            preds.append(t[i - 1])
            truth.append(t[i - 1 + horizon])
    return np.array(preds), np.array(truth)


def build_lstm(lookback, n_feat, units=64, dropout=0.2):
    inp = layers.Input(shape=(lookback, n_feat))
    x = layers.LSTM(units)(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    return Model(inp, out)


def fit_lstm(X_tr, y_tr, X_va, y_va, epochs=40, batch_size=256, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    m = build_lstm(X_tr.shape[1], X_tr.shape[2])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    m.fit(X_tr, y_tr, validation_data=(X_va, y_va),
          epochs=epochs, batch_size=batch_size, verbose=0,
          callbacks=[EarlyStopping(patience=6, restore_best_weights=True)])
    return m