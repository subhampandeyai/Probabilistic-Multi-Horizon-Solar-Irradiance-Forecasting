"""
Trains the Transformer, Informer-lite and TimesNet-lite reference models.

All three receive the same whole-series db4 level-3 decomposition and the same
backward-looking feature matrix as the proposed framework, under the same
chronological 70/15/15 split and seed 42, so the comparison isolates
architecture. Training uses validation early stopping.

Output: results/baselines/deep_baselines.csv  (station, horizon, model, r2)

The run is resumable: each (station, horizon, model) is appended as it finishes,
and completed triples are skipped on re-entry.

    python baselines/train_deep_baselines.py
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT.parent / "results" / "baselines"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = OUT / "deep_baselines.csv"

STATIONS = [1, 2, 4, 5, 6, 7, 8]
HORIZONS = [1, 4, 8, 16, 32, 96]
MODELS = ["Transformer", "Informer-lite", "TimesNet-lite"]

WAVELET, LEVEL = "db4", 3
SEQ_LEN = 24
SEED = 42

np.random.seed(SEED)


def decompose(signal):
    """Whole-series db4 level-3 bands, matching the proposed framework."""
    coeffs = pywt.wavedec(signal, WAVELET, level=LEVEL)
    bands = []
    for i in range(len(coeffs)):
        isolated = [np.zeros_like(c) for c in coeffs]
        isolated[i] = coeffs[i]
        bands.append(pywt.waverec(isolated, WAVELET)[: len(signal)])
    return np.column_stack(bands)


def build_features(df, signal, bands):
    """Lags, trailing rolling statistics, weather covariates and band signals."""
    n = len(signal)
    features = {}

    for lag in [1, 2, 4, 8, 16, 32]:
        col = np.full(n, np.nan)
        col[lag:] = signal[:-lag]
        features[f"lag{lag}"] = col

    series = pd.Series(signal)
    for window in [4, 12, 32]:
        features[f"roll_mean{window}"] = series.rolling(window, min_periods=1).mean().values
        features[f"roll_std{window}"] = series.rolling(window, min_periods=1).std().fillna(0).values

    for col in ["TEMPERATURE", "REL_HUMIDITY", "ATMOSPHERE", "DNI"]:
        if col in df.columns:
            features[col] = pd.to_numeric(df[col], errors="coerce").ffill().bfill().values

    for i in range(bands.shape[1]):
        features[f"band{i}"] = bands[:, i]

    return pd.DataFrame(features).values


def to_sequences(X, y, length):
    xs = [X[i - length + 1 : i + 1] for i in range(length - 1, len(X))]
    ys = [y[i] for i in range(length - 1, len(X))]
    return np.array(xs), np.array(ys)


def split(X, signal, horizon):
    """Chronological 70/15/15 on rows with a valid target and no missing features."""
    n = len(signal)
    y = np.full(n, np.nan)
    y[:-horizon] = signal[horizon:]
    keep = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    X, y = X[keep], y[keep]
    train_end, val_end = int(len(X) * 0.70), int(len(X) * 0.85)
    return (X[:train_end], y[:train_end],
            X[train_end:val_end], y[train_end:val_end],
            X[val_end:], y[val_end:])


def _keras():
    import tensorflow as tf

    tf.random.set_seed(SEED)
    from tensorflow.keras import Model, callbacks, layers

    return tf, Model, callbacks, layers


def _fit(model, xtr, ytr, xva, yva, xte, callbacks):
    model.compile("adam", "mse")
    model.fit(xtr, ytr, validation_data=(xva, yva), epochs=60, batch_size=128,
              verbose=0,
              callbacks=[callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
    return model.predict(xte, verbose=0).ravel(), SEQ_LEN - 1


def _prepare(Xtr, ytr, Xva, yva, Xte):
    scaler = StandardScaler()
    Xtr, Xva, Xte = scaler.fit_transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)
    xtr, ytr = to_sequences(Xtr, ytr, SEQ_LEN)
    xva, yva = to_sequences(Xva, yva, SEQ_LEN)
    xte, _ = to_sequences(Xte, np.zeros(len(Xte)), SEQ_LEN)
    return xtr, ytr, xva, yva, xte, Xtr.shape[1]


def transformer(Xtr, ytr, Xva, yva, Xte):
    """Two-layer encoder: 4 attention heads, model dimension 64, GELU expansion."""
    _, Model, callbacks, layers = _keras()
    xtr, ytr, xva, yva, xte, n_features = _prepare(Xtr, ytr, Xva, yva, Xte)

    inp = layers.Input((SEQ_LEN, n_features))
    x = layers.Dense(64)(inp)
    attention = layers.MultiHeadAttention(4, 16)(x, x)
    x = layers.LayerNormalization()(x + attention)
    feed_forward = layers.Dense(64, activation="gelu")(x)
    feed_forward = layers.Dense(64)(feed_forward)
    x = layers.LayerNormalization()(x + feed_forward)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(1)(x)

    return _fit(Model(inp, out), xtr, ytr, xva, yva, xte, callbacks)


def informer_lite(Xtr, ytr, Xva, yva, Xte):
    """Causal convolutional front end with distilling pooling, then attention."""
    _, Model, callbacks, layers = _keras()
    xtr, ytr, xva, yva, xte, n_features = _prepare(Xtr, ytr, Xva, yva, Xte)

    inp = layers.Input((SEQ_LEN, n_features))
    x = layers.Conv1D(64, 3, padding="causal", activation="gelu")(inp)
    x = layers.MaxPooling1D(2, padding="same")(x)
    attention = layers.MultiHeadAttention(4, 16)(x, x)
    x = layers.LayerNormalization()(x + attention)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(1)(x)

    return _fit(Model(inp, out), xtr, ytr, xva, yva, xte, callbacks)


def timesnet_lite(Xtr, ytr, Xva, yva, Xte):
    """Dilated causal convolutions capturing multi-period structure."""
    _, Model, callbacks, layers = _keras()
    xtr, ytr, xva, yva, xte, n_features = _prepare(Xtr, ytr, Xva, yva, Xte)

    inp = layers.Input((SEQ_LEN, n_features))
    x = layers.Conv1D(64, 3, padding="causal", activation="gelu")(inp)
    x = layers.Conv1D(64, 3, padding="causal", dilation_rate=2, activation="gelu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(1)(x)

    return _fit(Model(inp, out), xtr, ytr, xva, yva, xte, callbacks)


ARCHITECTURES = {
    "Transformer": transformer,
    "Informer-lite": informer_lite,
    "TimesNet-lite": timesnet_lite,
}


def completed():
    if RESULTS.exists():
        done = pd.read_csv(RESULTS)
        return set(zip(done.station, done.horizon, done.model))
    return set()


def main():
    done = completed()
    cache = {}
    started = time.time()

    for station in STATIONS:
        paths = list(PROCESSED.glob(f"station_{station:02d}_prepared.csv"))
        if not paths:
            print(f"  station {station}: no prepared file, skipping")
            continue

        if station not in cache:
            df = pd.read_csv(paths[0])
            signal = pd.to_numeric(df["IRRADIATION"], errors="coerce").ffill().bfill().values.astype(float)
            print(f"[decompose] station {station}")
            cache[station] = (signal, build_features(df, signal, decompose(signal)))

        signal, X = cache[station]

        for horizon in HORIZONS:
            Xtr, ytr, Xva, yva, Xte, yte = split(X, signal, horizon)
            if len(yte) < 100:
                continue

            for model in MODELS:
                if (station, f"H{horizon}", model) in done:
                    continue

                clock = time.time()
                try:
                    pred, offset = ARCHITECTURES[model](Xtr, ytr, Xva, yva, Xte)
                    r2 = r2_score(yte[offset:], pred)
                except Exception as exc:
                    print(f"  FAILED S{station} H{horizon} {model}: {exc}")
                    continue

                pd.DataFrame([{
                    "station": station,
                    "horizon": f"H{horizon}",
                    "model": model,
                    "r2": round(float(r2), 4),
                }]).to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)

                print(f"  S{station} H{horizon:<2} {model:<14} R2={r2:.4f} ({time.time() - clock:.0f}s)")

    print(f"\nComplete in {(time.time() - started) / 60:.1f} min -> {RESULTS}")


if __name__ == "__main__":
    main()
