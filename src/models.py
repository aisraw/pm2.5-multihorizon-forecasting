import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBRegressor


def make_windows(df, feats, target="pm25", lookback=24, horizon=1, flat=True):
    """Station-wise sliding window. flat=True -> XGBoost (2D), False -> deep (3D)."""
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


# ---------- deep models ----------

def build_lstm(lookback, n_feat, units=64, dropout=0.2):
    inp = layers.Input(shape=(lookback, n_feat))
    x = layers.LSTM(units)(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    return Model(inp, layers.Dense(1)(x), name="lstm")


def _attn_pool(seq, units=32):
    # Fix 2: tanh hidden layer diye attention score shikhbe
    score = layers.Dense(units, activation="tanh")(seq)
    score = layers.Dense(1)(score)                     # (B, L, 1)
    w = layers.Softmax(axis=1)(score)                  # time step er upor softmax
    ctx = layers.Multiply()([seq, w])
    return layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(ctx)


def build_lstm_attn(lookback, n_feat, units=64, dropout=0.2):
    inp = layers.Input(shape=(lookback, n_feat))
    seq = layers.LSTM(units, return_sequences=True)(inp)
    ctx = _attn_pool(seq)
    # Fix 4: last hidden state er sathe attention context concat
    last = layers.Lambda(lambda t: t[:, -1, :])(seq)
    x = layers.Concatenate()([ctx, last])
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    return Model(inp, layers.Dense(1)(x), name="lstm_attn")


def _encoder_block(x, d, heads, ff, dropout):
    a = layers.MultiHeadAttention(num_heads=heads, key_dim=d // heads,
                                  dropout=dropout)(x, x)
    x = layers.LayerNormalization()(layers.Add()([x, layers.Dropout(dropout)(a)]))
    f = layers.Dense(ff, activation="relu")(x)
    f = layers.Dense(d)(f)
    return layers.LayerNormalization()(layers.Add()([x, layers.Dropout(dropout)(f)]))


def build_lstm_transformer(lookback, n_feat, units=64, heads=4,
                           n_blocks=1, dropout=0.2):
    inp = layers.Input(shape=(lookback, n_feat))
    x = layers.LSTM(units, return_sequences=True)(inp)
    for _ in range(n_blocks):
        x = _encoder_block(x, units, heads, 2 * units, dropout)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    return Model(inp, layers.Dense(1)(x), name="lstm_transformer")


def fit_net(model, X_tr, y_tr, X_va, y_va, epochs=40, batch_size=256,
            seed=42, lr=3e-4, clip=1.0):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clip)
    model.compile(optimizer=opt, loss="huber")
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
              epochs=epochs, batch_size=batch_size, verbose=0,
              callbacks=[EarlyStopping(patience=6, restore_best_weights=True)])
    return model