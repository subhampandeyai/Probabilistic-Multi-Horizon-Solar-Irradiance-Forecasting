"""
Mean test R2 per model and horizon, aggregated across stations.

The stacked ensemble and the unified XGBoost reference are read from the Stage 3
station results; the remaining models are read from the trained runs under
results/baselines/. Only files carrying the measured schema (station, horizon,
model, r2) are read, so a model with no completed run is reported as missing
rather than filled in.

Outputs:
    results/model_comparison.csv
    results/model_comparison.tex

    python src/analysis/model_comparison.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STATION_RESULTS = ROOT / "results" / "station_results"
BASELINES = ROOT / "results" / "baselines"
OUT_CSV = ROOT / "results" / "model_comparison.csv"
OUT_TEX = ROOT / "results" / "model_comparison.tex"

STATIONS = [1, 2, 4, 5, 6, 7, 8]
HORIZONS = ["H1", "H4", "H8", "H16", "H32", "H96"]
COLUMNS = ["Proposed", "Unified XGB", "LightGBM",
           "Transformer", "Informer-lite", "TimesNet-lite"]


def from_station_results():
    """Mean proposed and unified R2 per horizon."""
    records = {s: json.load(open(STATION_RESULTS / f"station_{s:02d}_results.json"))
               for s in STATIONS}
    proposed, unified = {}, {}
    for horizon in HORIZONS:
        proposed[horizon] = sum(records[s][horizon]["fame_r2"] for s in STATIONS) / len(STATIONS)
        unified[horizon] = sum(records[s][horizon]["unified_r2"] for s in STATIONS) / len(STATIONS)
    return proposed, unified


def from_baseline_files():
    """Mean R2 per (model, horizon) across every baseline results file present.

    Only files carrying the measured schema (station, horizon, model, r2) are
    read, so nothing can enter the table except a trained run's own output.
    """
    schema = {"station", "horizon", "model", "r2"}
    frames = []
    for path in sorted(BASELINES.glob("*.csv")):
        frame = pd.read_csv(path)
        if schema.issubset(frame.columns):
            frames.append(frame[list(schema)])
    if not frames:
        return {}
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined.station.isin(STATIONS)]
    means = combined.groupby(["model", "horizon"])["r2"].mean()
    return {(model, horizon): value for (model, horizon), value in means.items()}


def main():
    proposed, unified = from_station_results()
    baselines = from_baseline_files()

    rows = []
    for horizon in HORIZONS:
        row = {"Horizon": horizon,
               "Proposed": proposed[horizon],
               "Unified XGB": unified[horizon]}
        for model in COLUMNS[2:]:
            row[model] = baselines.get((model, horizon))
        rows.append(row)

    table = pd.DataFrame(rows)
    table.loc[len(table)] = {"Horizon": "Mean",
                             **{c: table[c].mean() for c in COLUMNS}}

    missing = [c for c in COLUMNS if table[c].isna().any()]
    table.to_csv(OUT_CSV, index=False, float_format="%.4f")

    print(table.to_string(index=False, na_rep="MISSING", float_format=lambda v: f"{v:.4f}"))
    if missing:
        print("\nNot yet measured: " + ", ".join(missing))
        print("Run baselines/train_deep_baselines.py and baselines/train_lightgbm_baseline.py.")

    lines = ["\\begin{tabular}{l" + "r" * len(COLUMNS) + "}", "\\hline",
             "Horizon & " + " & ".join(COLUMNS) + " \\\\", "\\hline"]
    for _, row in table.iterrows():
        cells = [f"{row[c]:.4f}" if pd.notna(row[c]) else "---" for c in COLUMNS]
        lines.append(f"{row['Horizon']} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n-> {OUT_CSV}")
    print(f"-> {OUT_TEX}")


if __name__ == "__main__":
    main()
