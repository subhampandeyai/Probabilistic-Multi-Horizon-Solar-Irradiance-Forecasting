"""
Figure generation.

Reads the metrics written by the pipeline and analysis steps and produces:

    model_comparison_curve.pdf   performance frontier across horizons
    station_horizon_heatmap.pdf             station x horizon performance gain
    meta_coefficient_trajectories.pdf   meta-learner coefficient trajectories
    sample_day_forecasts.pdf         sample-day forecasts on representative days

Run the pipeline and the analysis steps first. sample_day_forecasts additionally needs the
stored test predictions from scripts/generate_sample_day_forecasts.py.

    python scripts/make_figures.py
"""
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
SUMMARY = RESULTS / "stage4_summary.csv"
BASELINES = RESULTS / "model_comparison.csv"
PRED_FP = RESULTS / "station_05_test_predictions.csv"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.linewidth': 0.9,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.edgecolor': 'gray',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'mathtext.fontset': 'stix',
})

COLORS = {
    'fame':        '#0B6E4F',
    'unified':     '#C44536',
    'lightgbm':    '#E69F00',
    'transformer': '#56B4E9',
    'informer':    '#9467bd',
    'timesnet':    '#8c564b',
    'persistence': '#888888',
    'trend':       '#0B6E4F',
    'daily':       '#D55E00',
    'hourly':      '#0072B2',
    'noise':       '#444444',
}

STATION_COLORS = {
    1: '#E41A1C', 2: '#377EB8', 4: '#4DAF4A', 5: '#984EA3',
    6: '#FF7F00', 7: '#A65628', 8: '#F781BF',
}
STATION_STYLES = {
    1: ('-', 'o'),  2: ('--', 's'),  4: (':', '^'),  5: ('-.', 'D'),
    6: ('-', 'v'),  7: ('--', 'p'),  8: (':', '*'),
}


# ============ DATA LOADERS ============
def load_stage4():
    df = pd.read_csv(SUMMARY)
    df = df[df['station'] != 3].copy()
    df['horizon'] = df['horizon'].astype(int)
    return df

def load_baselines():
    if not BASELINES.exists():
        print(f"  WARNING: {BASELINES} not found  -  model_comparison_curve will skip")
        return None
    return pd.read_csv(BASELINES)

def load_predictions():
    if not PRED_FP.exists():
        print(f"  WARNING: {PRED_FP} not found  -  sample_day_forecasts will skip")
        print(f"  Run scripts/generate_sample_day_forecasts.py first to enable sample_day_forecasts")
        return None
    return pd.read_csv(PRED_FP, parse_dates=['datetime'])


# ============ MODEL COMPARISON CURVE ============
def model_comparison_curve(df, baselines):
    if baselines is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5.2))
    horizons = sorted(df['horizon'].unique())
    fame_means = df.groupby('horizon')['fame_r2'].mean()
    fame_stds  = df.groupby('horizon')['fame_r2'].std()
    unif_means = df.groupby('horizon')['unified_r2'].mean()
    unif_stds  = df.groupby('horizon')['unified_r2'].std()

    bdf = baselines.copy()
    bdf['horizon_num'] = bdf['horizon'].str.replace('H', '').astype(int)

    for model, label, color, marker, ls in [
        ('informer_lite', 'Informer-lite', COLORS['informer'], 'v', '--'),
        ('transformer',   'Transformer',   COLORS['transformer'], 's', '--'),
        ('timesnet_lite', 'TimesNet-lite', COLORS['timesnet'], '^', '--'),
        ('lightgbm',      'LightGBM',      COLORS['lightgbm'], 'D', '-.'),
    ]:
        sub = bdf[bdf['model'] == model].groupby('horizon_num')['r2']
        means = sub.mean(); stds = sub.std()
        ax.plot(means.index, means.values, marker, linestyle=ls, color=color,
                label=label, linewidth=1.4, markersize=6, alpha=0.95, zorder=3)
        ax.fill_between(means.index, means.values - stds.values, means.values + stds.values,
                        alpha=0.10, color=color, zorder=2)

    ax.plot(horizons, unif_means.values, 's-', color=COLORS['unified'],
            label='Unified XGBoost', linewidth=2.0, markersize=7, zorder=4)
    ax.fill_between(horizons, unif_means - unif_stds, unif_means + unif_stds,
                    alpha=0.15, color=COLORS['unified'])

    ax.plot(horizons, fame_means.values, 'o-', color=COLORS['fame'],
            label='FAME (proposed)', linewidth=2.6, markersize=9, zorder=5)
    ax.fill_between(horizons, fame_means - fame_stds, fame_means + fame_stds,
                    alpha=0.20, color=COLORS['fame'])

    transf_h96 = bdf[(bdf['model']=='transformer') & (bdf['horizon_num']==96)]['r2'].mean()
    fame_h96 = fame_means[96]
    ax.annotate(f'+{(fame_h96 - transf_h96)*100:.1f} pp\nvs Transformer',
                xy=(96, transf_h96), xytext=(48, 0.30),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', lw=0.6))

    ax.set_xscale('log')
    ax.set_xticks(horizons)
    ax.set_xticklabels([f'H{h}' for h in horizons])
    ax.set_xlabel('Forecast horizon (steps; H1 = 15 min, H96 = 24 h)', fontsize=11)
    ax.set_ylabel(r'Mean test $R^{2}$ across 7 stations', fontsize=11)
    ax.set_ylim(0.05, 1.02)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc='lower left', ncol=2, fontsize=9)
    plt.tight_layout()
    out = OUT_DIR / "model_comparison_curve.pdf"
    plt.savefig(out); plt.savefig(str(out).replace('.pdf', '.png'), dpi=300)
    plt.close()
    print(f"  [saved] {out.name}")


# ============ STATION x HORIZON HEATMAP ============
def station_horizon_heatmap(df):
    pivot = df.pivot(index='station', columns='horizon', values='improvement_pp')
    pivot = pivot.reindex(sorted(pivot.index)).reindex(columns=sorted(pivot.columns))
    fig = plt.figure(figsize=(9, 5.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[6, 1], height_ratios=[1, 6],
                          hspace=0.05, wspace=0.05)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    cmap = LinearSegmentedColormap.from_list("rwb",
        ["#9d2727", "#e57373", "white", "#64b5f6", "#0d47a1"], N=256)
    norm = Normalize(vmin=-5, vmax=5)
    im = ax_main.imshow(pivot.values, cmap=cmap, norm=norm, aspect='auto')

    for i, _ in enumerate(pivot.index):
        for j, _ in enumerate(pivot.columns):
            v = pivot.iloc[i, j]
            color = 'white' if abs(v) > 3 else 'black'
            ax_main.text(j, i, f'{v:+.2f}', ha='center', va='center',
                         fontsize=10, color=color,
                         fontweight='bold' if abs(v) > 2 else 'normal')

    ax_main.set_xticks(range(len(pivot.columns)))
    ax_main.set_xticklabels([f'H{h}' for h in pivot.columns], fontsize=10)
    ax_main.set_yticks(range(len(pivot.index)))
    ax_main.set_yticklabels([f'Stn {s}' for s in pivot.index], fontsize=10)
    ax_main.set_xlabel('Forecast horizon', fontsize=11)
    ax_main.set_ylabel('Station', fontsize=11)

    col_means = pivot.mean(axis=0)
    ax_top.bar(range(len(col_means)), col_means.values,
               color=[cmap(norm(v)) for v in col_means.values],
               edgecolor='black', linewidth=0.4)
    ax_top.axhline(0, color='black', linewidth=0.6)
    ax_top.set_ylabel('Mean delta \n(pp)', fontsize=9)
    ax_top.tick_params(labelbottom=False, labelsize=8)
    for i, v in enumerate(col_means.values):
        ax_top.text(i, v + 0.3 if v >= 0 else v - 0.5, f'{v:+.2f}',
                    ha='center', fontsize=8)

    row_means = pivot.mean(axis=1)
    ax_right.barh(range(len(row_means)), row_means.values,
                  color=[cmap(norm(v)) for v in row_means.values],
                  edgecolor='black', linewidth=0.4)
    ax_right.axvline(0, color='black', linewidth=0.6)
    ax_right.set_xlabel('Mean delta  (pp)', fontsize=9)
    ax_right.tick_params(labelleft=False, labelsize=8)
    for i, v in enumerate(row_means.values):
        ax_right.text(v + 0.1 if v >= 0 else v - 0.4, i, f'{v:+.2f}',
                      va='center', fontsize=8)

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.55])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r'$\Delta R^{2}$ FAME $-$ unified XGB (pp)', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    plt.suptitle(f'Station x horizon performance gain  (overall mean +{pivot.values.mean():.2f} pp, n=42)',
                 fontsize=11, y=0.96)
    out = OUT_DIR / "station_horizon_heatmap.pdf"
    plt.savefig(out); plt.savefig(str(out).replace('.pdf', '.png'), dpi=300)
    plt.close()
    print(f"  [saved] {out.name}")


# ============ META-LEARNER COEFFICIENTS ============
def meta_coefficient_trajectories(df):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    horizons = sorted(df['horizon'].unique())
    stations = sorted(df['station'].unique())

    bands = [
        ('weight_trend',  COLORS['trend'],  'Trend (Ridge)  -  slow envelope',         axes[0, 0]),
        ('weight_daily',  COLORS['daily'],  'Daily (LSTM)  -  1-2 h variation',        axes[0, 1]),
        ('weight_hourly', COLORS['hourly'], 'Hourly (XGBoost)  -  30-60 min',          axes[1, 0]),
        ('weight_noise',  COLORS['noise'],  'Noise (Persistence)  -  15-30 min',       axes[1, 1]),
    ]

    for col, mean_color, label, ax in bands:
        for s in stations:
            sub = df[df['station'] == s].sort_values('horizon')
            ls, mk = STATION_STYLES[s]
            ax.plot(sub['horizon'], sub[col],
                    color=STATION_COLORS[s], linestyle=ls, marker=mk,
                    linewidth=1.6, markersize=6, alpha=0.85,
                    markeredgecolor='white', markeredgewidth=0.6,
                    label=f'Stn {s}' if col == 'weight_trend' else None,
                    zorder=3)

        mean = df.groupby('horizon')[col].mean()
        std = df.groupby('horizon')[col].std()
        ax.fill_between(horizons, mean - std, mean + std, alpha=0.18,
                        color=mean_color, zorder=4)
        ax.plot(horizons, mean.values, color=mean_color, linewidth=3.5,
                marker='o', markersize=9, markeredgecolor='white',
                markeredgewidth=1.2, zorder=5,
                label='Cross-station mean +/-1sigma' if col == 'weight_trend' else None)

        ax.axhline(0, color='black', linewidth=0.6, linestyle='-', alpha=0.6)
        ax.set_xscale('log')
        ax.set_xticks(horizons)
        ax.set_xticklabels([f'H{h}' for h in horizons])
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_ylabel(r'Meta-learner coefficient $\beta_k$', fontsize=10)

    axes[1, 0].set_xlabel('Forecast horizon', fontsize=11)
    axes[1, 1].set_xlabel('Forecast horizon', fontsize=11)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if 'Cross-station mean +/-1sigma' in labels:
        idx = labels.index('Cross-station mean +/-1sigma')
        handles = [handles[idx]] + [h for i, h in enumerate(handles) if i != idx]
        labels = [labels[idx]] + [l for i, l in enumerate(labels) if i != idx]
    axes[0, 0].legend(handles, labels, loc='upper right', ncol=2, fontsize=8.5,
                       columnspacing=0.8, handlelength=2.2, handletextpad=0.5)

    plt.suptitle('Meta-learner coefficient trajectories across horizons (all 7 stations)',
                 fontsize=13, fontweight='bold', y=1.0)
    plt.tight_layout()
    out = OUT_DIR / "meta_coefficient_trajectories.pdf"
    plt.savefig(out); plt.savefig(str(out).replace('.pdf', '.png'), dpi=300)
    plt.close()
    print(f"  [saved] {out.name}")


# ============ SAMPLE-DAY FORECASTS ============
def sample_day_forecasts(pred_df):
    if pred_df is None:
        return
    pred_df = pred_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(pred_df['datetime']):
        pred_df['datetime'] = pd.to_datetime(pred_df['datetime'], errors='coerce')
    pred_df = pred_df.dropna(subset=['datetime'])
    pred_df['date'] = pred_df['datetime'].dt.date
    pred_df['hour'] = pred_df['datetime'].dt.hour + pred_df['datetime'].dt.minute / 60.0

    # Pick three days with distinct cloud regimes:
    #   (a) clear: lowest day-level std relative to mean
    #   (b) cloudy: middle range
    #   (c) variable: highest std/mean ratio
    by_day = pred_df.groupby('date').agg(
        mean_obs=('observed', 'mean'),
        std_obs=('observed', 'std'),
        n=('observed', 'count'),
    )
    by_day = by_day[by_day['n'] >= 30]  # need full day
    by_day = by_day[by_day['mean_obs'] > 0.05]  # exclude winter
    by_day['cv'] = by_day['std_obs'] / by_day['mean_obs']
    by_day = by_day.sort_values('cv')

    if len(by_day) < 3:
        print("  not enough complete test days, skipping sample_day_forecasts")
        return

    clear_day  = by_day.iloc[len(by_day) // 10].name           # low CV
    cloudy_day = by_day.iloc[len(by_day) // 2].name             # median CV
    variable_day = by_day.iloc[-len(by_day) // 10].name         # high CV
    chosen = [(clear_day, 'Clear day'), (cloudy_day, 'Cloudy day'),
              (variable_day, 'Variable day')]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (day, title) in zip(axes, chosen):
        d = pred_df[pred_df['date'] == day].sort_values('hour')
        ax.plot(d['hour'], d['observed'], '-', color='black',
                linewidth=2.0, label='Observed', zorder=5)
        ax.plot(d['hour'], d['fame'], '-', color=COLORS['fame'],
                linewidth=1.5, alpha=0.9, label='FAME', zorder=4)
        ax.plot(d['hour'], d['unified_xgb'], '--', color=COLORS['unified'],
                linewidth=1.2, alpha=0.85, label='Unified XGB', zorder=3)
        if 'persistence' in d.columns:
            ax.plot(d['hour'], d['persistence'], '-.', color=COLORS['persistence'],
                    linewidth=1.0, alpha=0.7, label='Persistence', zorder=1)
        # Compute MAE on this day for annotation
        mae_fame = float(np.mean(np.abs(d['observed'] - d['fame'])))
        mae_unif = float(np.mean(np.abs(d['observed'] - d['unified_xgb'])))
        ax.text(0.02, 0.97,
                f'MAE FAME: {mae_fame:.1f} W/m^2\nMAE Unif: {mae_unif:.1f} W/m^2',
                transform=ax.transAxes, fontsize=8.5, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', lw=0.5))
        ax.set_xlabel('Hour of day', fontsize=10)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_title(f'{title} ({day})', fontsize=10)

    axes[0].set_ylabel(r'GHI (W/m$^{2}$)', fontsize=10)
    axes[0].legend(loc='upper left', fontsize=8.5, ncol=1)
    plt.suptitle('Sample H1 forecasts on three representative test days (Station 5)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "sample_day_forecasts.pdf"
    plt.savefig(out); plt.savefig(str(out).replace('.pdf', '.png'), dpi=300)
    plt.close()
    print(f"  [saved] {out.name}")


# ============ MAIN ============
def main():
    print("=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    print("\n--- Loading data ---")
    df = load_stage4(); print(f"  stage4 rows: {len(df)}")
    baselines = load_baselines()
    if baselines is not None: print(f"  baseline rows: {len(baselines)}")
    pred_df = load_predictions()

    print("\n--- Building figures ---")
    for fn, name in [
        (lambda: model_comparison_curve(df, baselines), 'model_comparison_curve'),
        (lambda: station_horizon_heatmap(df), 'station_horizon_heatmap'),
        (lambda: meta_coefficient_trajectories(df), 'meta_coefficient_trajectories'),
        (lambda: sample_day_forecasts(pred_df), 'sample_day_forecasts'),
    ]:
        print(f"\n[{name}]")
        try: fn()
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Figures written to {OUT_DIR}")
    print("=" * 70)
    print("\nsample_day_forecasts requires the stored test predictions; generate them with:")
    print("    python scripts/generate_sample_day_forecasts.py")
    print("    then re-run this script.")

if __name__ == "__main__":
    main()




