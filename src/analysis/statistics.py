"""
Paired significance tests between the stacked ensemble and the reference model.

Computes the one-sided Wilcoxon signed-rank test on the paired per-station R2
differences, Cohen's d for effect size, and the Holm-Bonferroni correction
across the per-horizon tests. The Wilcoxon test is used because the paired
differences are not assumed normal across stations.

Outputs:
    results/statistics.csv   per-horizon and aggregate results
    results/statistics.tex

    python src/analysis/statistics.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
STATION_RESULTS = ROOT / "results" / "station_results"
OUT_CSV = ROOT / "results" / "statistics.csv"
OUT_TEX = ROOT / "results" / "statistics.tex"

STATIONS = [1, 2, 4, 5, 6, 7, 8]
HORIZONS = ["H1", "H4", "H8", "H16", "H32", "H96"]


def load_differences():
    """Paired proposed-minus-unified R2 differences, keyed by horizon."""
    differences = {}
    for horizon in HORIZONS:
        paired = []
        for station in STATIONS:
            path = STATION_RESULTS / f"station_{station:02d}_results.json"
            record = json.load(open(path))[horizon]
            paired.append(record["fame_r2"] - record["unified_r2"])
        differences[horizon] = np.array(paired)
    return differences


def cohens_d(differences):
    """Effect size: mean paired difference over its standard deviation."""
    return float(differences.mean() / differences.std(ddof=1))


def holm_bonferroni(p_values):
    """Step-down correction; returns adjusted p-values in the input order."""
    order = np.argsort(p_values)
    n = len(p_values)
    adjusted = np.empty(n)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (n - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def main():
    differences = load_differences()

    rows = []
    raw_p = []
    for horizon in HORIZONS:
        paired = differences[horizon]
        _, p = stats.wilcoxon(paired, alternative="greater")
        raw_p.append(p)
        rows.append({
            "horizon": horizon,
            "n_stations": len(paired),
            "mean_delta_pp": round(float(paired.mean() * 100), 4),
            "wins": int((paired > 0).sum()),
            "cohens_d": round(cohens_d(paired), 4),
            "wilcoxon_p": round(float(p), 6),
        })

    for row, adjusted in zip(rows, holm_bonferroni(np.array(raw_p))):
        row["holm_p"] = round(float(adjusted), 6)
        row["significant_at_0.05"] = bool(adjusted < 0.05)

    pooled = np.concatenate([differences[h] for h in HORIZONS])
    _, aggregate_p = stats.wilcoxon(pooled, alternative="greater")
    rows.append({
        "horizon": "aggregate",
        "n_stations": len(pooled),
        "mean_delta_pp": round(float(pooled.mean() * 100), 4),
        "wins": int((pooled > 0).sum()),
        "cohens_d": round(cohens_d(pooled), 4),
        "wilcoxon_p": float(f"{aggregate_p:.3e}"),
        "holm_p": None,
        "significant_at_0.05": bool(aggregate_p < 0.05),
    })

    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False)
    print(table.to_string(index=False))
    print(f"\n-> {OUT_CSV}")

    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\hline",
        "Horizon & Wins & $\\Delta R^2$ (pp) & Cohen's $d$ & Wilcoxon $p$ & Holm $p$ \\\\",
        "\\hline",
    ]
    for row in rows[:-1]:
        lines.append(
            f"{row['horizon']} & {row['wins']}/{row['n_stations']} & "
            f"{row['mean_delta_pp']:+.2f} & {row['cohens_d']:.2f} & "
            f"{row['wilcoxon_p']:.4f} & {row['holm_p']:.4f} \\\\"
        )
    final = rows[-1]
    lines += [
        "\\hline",
        f"All {final['n_stations']} pairs & {final['wins']}/{final['n_stations']} & "
        f"{final['mean_delta_pp']:+.2f} & {final['cohens_d']:.2f} & "
        f"{final['wilcoxon_p']:.2e} & --- \\\\",
        "\\hline",
        "\\end{tabular}",
    ]
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {OUT_TEX}")


if __name__ == "__main__":
    main()
