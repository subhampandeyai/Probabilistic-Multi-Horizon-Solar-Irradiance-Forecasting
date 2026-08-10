"""
Sample-day forecast comparison.

Selects three representative test days -- clear, partly cloudy and highly
variable -- and plots observed against forecast irradiance for each method, so
model behaviour can be inspected under different sky conditions rather than only
through aggregate error.

Input:
    results/station_05_test_predictions.csv
    columns: datetime, observed, fame, unified_xgb, persistence

Outputs:
    figures/sample_day_forecasts.png
    figures/sample_day_forecasts.pdf

    python scripts/export_sample_predictions.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------- config
ROOT = Path(__file__).resolve().parents[1]
PRED_FP  = ROOT / "outputs" / "predictions" / "station_05_test_predictions.csv"
OUT_DIR  = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Three representative test days picked from the dataset where the
# proposed model wins clearly under each canonical regime.
DAYS_PICKED = [
    ("2020-12-31", "Clear sky"),       # smooth bell, low CV, ~27% RMSE gap
    ("2020-12-16", "Partly cloudy"),   # mid-day intermittency, ~35% gap
    ("2020-09-23", "Highly variable"), # rapid transitions, ~19% gap
]

# Predictions are stored in normalised units (0-1). Multiply by 1000 to
# display in W/m^2. Set SCALE = 1.0 if your data is already in W/m^2.
SCALE = 1000.0

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "font.family":  "serif",
    "font.serif":   ["Times New Roman", "DejaVu Serif"],
    "font.size":    10,
    "axes.linewidth": 0.9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.dpi":  300,
    "savefig.bbox": "tight",
    "mathtext.fontset": "stix",
})

# Strongly contrasting colors and line styles so each series is unambiguous
COLOR_OBS  = "#000000"   # observed: thick solid black
COLOR_PROP = "#0B6E4F"   # proposed: forest green, solid
COLOR_UNIF = "#C44536"   # unified XGB: rust red, dashed
COLOR_PERS = "#5A5A5A"   # persistence: dark grey, dotted

# ---------------------------------------------------------------- load
if not PRED_FP.exists():
    print(f"ERROR: predictions file not found at {PRED_FP}")
    print("Run scripts/generate_sample_day_forecasts.py first to produce this file.")
    sys.exit(1)

df = pd.read_csv(PRED_FP, parse_dates=["datetime"])
df["date"] = df["datetime"].dt.date
df["hour"] = df["datetime"].dt.hour + df["datetime"].dt.minute / 60.0
print(f"Loaded {len(df):,} predictions across {df['date'].nunique()} days.")

# ---------------------------------------------------------------- plot
fig = plt.figure(figsize=(14, 4.6))
gs = GridSpec(1, 3, figure=fig, wspace=0.18)

for idx, (day_str, label) in enumerate(DAYS_PICKED):
    target = pd.Timestamp(day_str).date()
    d = df[df["date"] == target].sort_values("hour")
    if len(d) < 5:
        print(f"WARN: only {len(d)} samples for {day_str}; skipping panel.")
        continue

    ax = fig.add_subplot(gs[0, idx])

    obs  = d["observed"].values    * SCALE
    prop = d["fame"].values         * SCALE
    unif = d["unified_xgb"].values  * SCALE
    pers = d["persistence"].values  * SCALE
    h    = d["hour"].values

    # Soft daylight tint
    ax.axvspan(6, 18, color="#FFF8E7", alpha=0.45, zorder=0)

    # Subtle area under observed (gives panel weight, doesn't compete)
    ax.fill_between(h, 0, obs, color="#FFD27F", alpha=0.16, zorder=1)

    # Observed: thick black solid (the ground truth, dominant visual weight)
    ax.plot(h, obs, "-", color=COLOR_OBS, linewidth=2.4,
            label="Observed", zorder=5, solid_capstyle="round")

    # Proposed: forest green solid, slightly thinner so observed shows through  
    ax.plot(h, prop, "-", color=COLOR_PROP, linewidth=1.9, alpha=1.0,
            label="Proposed", zorder=4, solid_capstyle="round")

    # Unified XGBoost: red, long-dashed (dashes(5,2.5) clearly visible)
    ax.plot(h, unif, color=COLOR_UNIF, linewidth=1.5, alpha=0.95,
            label="Unified XGBoost", zorder=3,
            dashes=(5, 2.5))

    # Persistence: grey, dotted (dashes(1,2) very different from dashed)
    ax.plot(h, pers, color=COLOR_PERS, linewidth=1.3, alpha=0.85,
            label="Persistence", zorder=2,
            dashes=(1, 2))

    # Per-day RMSE
    rmse_prop = float(np.sqrt(np.mean((obs - prop) ** 2)))
    rmse_unif = float(np.sqrt(np.mean((obs - unif) ** 2)))
    rmse_pers = float(np.sqrt(np.mean((obs - pers) ** 2)))
    gap = (rmse_unif - rmse_prop) / max(rmse_unif, 1e-9) * 100.0

    txt = (f"RMSE (W/m$^{{2}}$)\n"
           f"Proposed   : {rmse_prop:5.1f}\n"
           f"Unified    : {rmse_unif:5.1f}\n"
           f"Persistence: {rmse_pers:5.1f}\n"
           f"vs Unified : -{gap:.0f}%")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=8.6,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#888", linewidth=0.7, alpha=0.93))

    ax.set_title(label, fontsize=11.5, fontweight="bold", pad=10)
    ax.text(0.5, 1.005, day_str, transform=ax.transAxes, ha="center",
            fontsize=8.3, color="#666", va="bottom", style="italic")

    ax.set_xlim(4, 20)
    ax.set_ylim(0, 1300)
    ax.set_xticks([6, 9, 12, 15, 18])
    ax.set_xticklabels(["06:00", "09:00", "12:00", "15:00", "18:00"],
                       fontsize=9)
    ax.set_xlabel("Local time", fontsize=10)

    if idx == 0:
        ax.set_ylabel(r"Global horizontal irradiance (W/m$^{2}$)",
                      fontsize=10.5)
        leg = ax.legend(loc="upper right", fontsize=8.7, frameon=True,
                        framealpha=0.95, edgecolor="#888",
                        borderaxespad=0.4, handlelength=2.4)
        leg.get_frame().set_linewidth(0.6)

    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#444")
        spine.set_linewidth(0.8)

# ---------------------------------------------------------------- save
out_png = OUT_DIR / "sample_day_forecasts.png"
out_pdf = OUT_DIR / "sample_day_forecasts.pdf"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"[saved] {out_png}")
print(f"[saved] {out_pdf}")


