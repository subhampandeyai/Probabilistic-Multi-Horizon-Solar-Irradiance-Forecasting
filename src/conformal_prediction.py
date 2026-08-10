"""
Split conformal prediction intervals.

Wraps stored point forecasts in distribution-free prediction intervals. The
guarantee is finite-sample and holds without any assumption on the residual
distribution, provided the calibration and evaluation samples are exchangeable.

Procedure:
    1. Split the test predictions chronologically into a calibration half and an
       evaluation half.
    2. Score the calibration half by nonconformity s_i = |y_i - yhat_i|.
    3. Take q_alpha as the ceil((n+1)(1-alpha))-th order statistic of the scores.
    4. Form [yhat - q_alpha, yhat + q_alpha] on the evaluation half.
    5. Report empirical coverage and mean interval width at each nominal level.

Coverage and width are reported together: an interval that is wide enough to
always cover is not informative, so neither number is meaningful alone.

Input:
    results/station_05_test_predictions.csv

Outputs:
    results/conformal_coverage.csv     coverage and width per method and level
    results/conformal_intervals.csv    per-sample intervals
    figures/prediction_intervals.png

    python src/conformal_prediction.py
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
os.chdir(ROOT)
PROJECT = ROOT.parent

PRED_FP = PROJECT / "results" / "station_05_test_predictions.csv"
OUT_DIR = PROJECT / "results"
FIG_DIR = PROJECT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

if not PRED_FP.exists():
    print(f"ERROR: {PRED_FP} not found.")
    print("Generate it with: python scripts/export_sample_predictions.py")
    sys.exit(1)

df = pd.read_csv(PRED_FP)
print(f"Loaded {len(df)} rows from {PRED_FP.name}")
print(f"Columns: {list(df.columns)}")

SCALE = 1000.0
df["observed_W"]   = df["observed"]    * SCALE
df["fame_W"]       = df["fame"]        * SCALE
df["unified_W"]    = df["unified_xgb"] * SCALE
df["persist_W"]    = df["persistence"] * SCALE

# Daytime filter (obs > 50 W/m^2)
day = df[df["observed_W"] > 50].copy().reset_index(drop=True)
print(f"Daytime rows: {len(day)}")

# Split into calibration + test (first half / second half, chronological)
n = len(day)
n_cal = n // 2
cal = day.iloc[:n_cal].copy()
test = day.iloc[n_cal:].copy()
print(f"Calibration: {len(cal)} rows | Test: {len(test)} rows")

# Confidence levels to evaluate
ALPHAS = [0.10, 0.05]  # corresponds to 90% and 95% nominal coverage

rows = []
intervals_records = []

for method, col in [("FAME", "fame_W"), ("Unified XGB", "unified_W"), ("Persistence", "persist_W")]:
    # Nonconformity scores = absolute residuals on calibration set
    cal_pred = cal[col].values
    cal_obs  = cal["observed_W"].values
    scores   = np.abs(cal_obs - cal_pred)

    test_pred = test[col].values
    test_obs  = test["observed_W"].values

    print(f"\n--- {method} ---")
    for alpha in ALPHAS:
        nominal = 1 - alpha
        # Conformal quantile (with finite-sample correction)
        q_index = int(np.ceil((len(scores) + 1) * (1 - alpha))) - 1
        q_index = min(q_index, len(scores) - 1)
        q_alpha = np.sort(scores)[q_index]

        lower = test_pred - q_alpha
        upper = test_pred + q_alpha
        in_interval = (test_obs >= lower) & (test_obs <= upper)
        empirical_cov = float(in_interval.mean())
        mean_width = float((upper - lower).mean())

        rows.append({
            "method": method,
            "nominal_coverage_%": int(nominal * 100),
            "empirical_coverage_%": round(empirical_cov * 100, 2),
            "q_alpha_W_m2": round(q_alpha, 2),
            "mean_interval_width_W_m2": round(mean_width, 2),
            "n_test": int(len(test_obs)),
        })
        print(f"  nominal {int(nominal*100)}%: empirical={empirical_cov*100:.2f}%  q_alpha={q_alpha:.2f} W/m^2  width={mean_width:.2f}")

        if method == "FAME":
            for i, (lo, pr, hi, ob) in enumerate(zip(lower, test_pred, upper, test_obs)):
                intervals_records.append({
                    "datetime": test.iloc[i]["datetime"] if "datetime" in test.columns else i,
                    "alpha": alpha,
                    "lower_W_m2": round(lo, 2),
                    "pred_W_m2": round(pr, 2),
                    "upper_W_m2": round(hi, 2),
                    "obs_W_m2": round(ob, 2),
                })

# Save coverage CSV
cov_df = pd.DataFrame(rows)
cov_df.to_csv(OUT_DIR / "conformal_coverage.csv", index=False)
print(f"\nSaved coverage stats -> results/conformal_coverage.csv")

# Save per-sample intervals (FAME only)
int_df = pd.DataFrame(intervals_records)
int_df.to_csv(OUT_DIR / "conformal_intervals.csv", index=False)
print(f"Saved per-sample intervals -> results/conformal_intervals.csv")

# Plot: forecast + 90% interval on a sample window
print("\nGenerating coverage plot...")
fame_intervals = int_df[int_df["alpha"] == 0.10].reset_index(drop=True)
plot_n = min(300, len(fame_intervals))  # show first 300 test points (~3 days of 15-min data)
sub = fame_intervals.iloc[:plot_n]

fig, ax = plt.subplots(figsize=(11, 4.5), dpi=120)
x = np.arange(plot_n)
ax.fill_between(x, sub["lower_W_m2"], sub["upper_W_m2"],
                 alpha=0.25, color="#3b7ab5", label="90% Conformal interval")
ax.plot(x, sub["pred_W_m2"], color="#1a5490", lw=1.5, label="FAME forecast")
ax.plot(x, sub["obs_W_m2"],  color="#cc4400", lw=1.0, marker=".", ms=3, label="Observed")
ax.set_xlabel("Test sample index (15-min steps, Station 5, $H_{1}$)")
ax.set_ylabel("GHI (W/m$^{2}$)")
ax.set_title("Conformal prediction intervals: FAME forecast with 90% coverage on Station 5, $H_{1}$")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "prediction_intervals.png", dpi=150, bbox_inches="tight")
plt.savefig(FIG_DIR / "prediction_intervals.pdf", bbox_inches="tight")
plt.close()
print(f"Plot -> figures/prediction_intervals.png and .pdf")

print("\n" + "="*60)
print("CONFORMAL PREDICTION ANALYSIS COMPLETE")
print("="*60)
