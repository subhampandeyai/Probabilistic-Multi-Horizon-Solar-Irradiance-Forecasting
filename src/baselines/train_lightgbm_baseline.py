"""
Trains the LightGBM reference model.

Uses the same whole-series db4 level-3 decomposition, feature matrix, split and
seed as train_deep_baselines.py, so the reference models are directly
comparable: 500 trees, depth 8, learning rate 0.05.

Output: results/baselines/lightgbm.csv  (station, horizon, model, r2)

    python baselines/train_lightgbm_baseline.py
"""
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import r2_score

from train_deep_baselines import (HORIZONS, PROCESSED, SEED, STATIONS,
                                  build_features, decompose, split)

warnings.filterwarnings("ignore")

RESULTS = Path(__file__).resolve().parents[2] / "results" / "baselines" / "lightgbm.csv"
RESULTS.parent.mkdir(parents=True, exist_ok=True)


def main():
    started = time.time()
    rows = []

    for station in STATIONS:
        paths = list(PROCESSED.glob(f"station_{station:02d}_prepared.csv"))
        if not paths:
            print(f"  station {station}: no prepared file, skipping")
            continue

        df = pd.read_csv(paths[0])
        signal = pd.to_numeric(df["IRRADIATION"], errors="coerce").ffill().bfill().values.astype(float)
        X = build_features(df, signal, decompose(signal))

        for horizon in HORIZONS:
            Xtr, ytr, Xva, yva, Xte, yte = split(X, signal, horizon)
            if len(yte) < 100:
                continue

            model = lgb.LGBMRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                random_state=SEED,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])

            r2 = r2_score(yte, model.predict(Xte))
            rows.append({
                "station": station,
                "horizon": f"H{horizon}",
                "model": "LightGBM",
                "r2": round(float(r2), 4),
            })
            print(f"  S{station} H{horizon:<2} LightGBM R2={r2:.4f}")

    pd.DataFrame(rows).to_csv(RESULTS, index=False)
    print(f"\nComplete in {(time.time() - started) / 60:.1f} min -> {RESULTS}")


if __name__ == "__main__":
    main()
