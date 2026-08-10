"""
Meta-learner coefficient trajectories across forecast horizons.

Reads the Ridge meta-learner weights stored by Stage 3 for each station and
horizon and plots the four band trajectories with a cross-station mean and a
+/-1 standard-deviation band. The coefficients show how much each frequency band
contributes to the fused forecast as lead time grows.

The weights are only interpretable when each band ran a distinct learner; Stage 3
records the learner actually used per band so a run with a substituted learner
can be identified.

Outputs:
    results/meta_coefficients.csv
    figures/meta_coefficients.png

    python src/analysis/meta_coefficients.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

STATIONS = [1, 2, 4, 5, 6, 7, 8]
HORIZONS = ["H1", "H4", "H8", "H16", "H32", "H96"]
BANDS = ["trend", "daily", "hourly", "noise"]
BAND_LABEL = {
    "trend":  "Trend (Ridge) - slow envelope",
    "daily":  "Daily (LSTM) - 1-2 h variation",
    "hourly": "Hourly (XGBoost) - 30-60 min",
    "noise":  "Noise (Persistence) - 15-30 min",
}


def load():
    """Collect the stored meta-learner weights for every station and horizon."""
    rows = []
    for station in STATIONS:
        path = RESULTS / "station_results" / f"station_{station:02d}_results.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        for horizon in HORIZONS:
            weights = (record.get(horizon) or {}).get("meta_weights")
            if weights:
                rows.append({"station": station, "horizon": horizon,
                             **{b: weights.get(b) for b in BANDS}})
    return pd.DataFrame(rows)


def plot(df):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    x = range(len(HORIZONS))
    colours = plt.cm.tab10([i / len(STATIONS) for i in range(len(STATIONS))])

    for ax, band in zip(axes.ravel(), BANDS):
        for colour, station in zip(colours, STATIONS):
            series = (df[df.station == station]
                      .set_index("horizon").reindex(HORIZONS)[band])
            ax.plot(x, series, marker="o", ms=4, lw=1.1, alpha=0.65,
                    color=colour, label=f"Stn {station}")
        mean = df.groupby("horizon")[band].mean().reindex(HORIZONS)
        std = df.groupby("horizon")[band].std(ddof=1).reindex(HORIZONS)
        ax.plot(x, mean, lw=3, color="black", label="Cross-station mean", zorder=5)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color="grey")
        ax.axhline(0, color="0.4", lw=0.8, ls="--")
        ax.set_title(BAND_LABEL[band], fontsize=11)
        ax.set_xticks(list(x))
        ax.set_xticklabels(HORIZONS)
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel(r"Meta-learner coefficient $\beta$")
        ax.grid(alpha=0.25)

    axes[0, 0].legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle("Meta-learner coefficient trajectories across horizons\n"
                 "(all 7 stations, 4 frequency bands)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIGURES / "meta_coefficients.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    df = load()
    if df.empty:
        print("No meta-learner weights found. Run scripts/run_pipeline.py first.")
        return

    df.to_csv(RESULTS / "meta_coefficients.csv", index=False,
              float_format="%.6f")

    print("Cross-station mean coefficients by horizon")
    print("=" * 60)
    print(df.groupby("horizon")[BANDS].mean().reindex(HORIZONS)
          .to_string(float_format=lambda v: f"{v:+.4f}"))

    out = plot(df)
    print(f"\n-> {RESULTS / 'meta_coefficients.csv'}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
