"""
Stage 4: cross-station evaluation.

Aggregates the per-station Stage 3 results into a single summary, runs the
paired significance tests across stations, and renders the diagnostic plots.

Input:  results/station_results/station_XX_results.json
Output: results/stage4_evaluation.json
        results/stage4_summary.csv
        figures/stage4/*.png

    python -m src.pipeline.stage4_evaluation
"""
import os, sys, glob, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import cfg, get_path, set_all_seeds
from utils.metrics import save_metrics
from utils.plotting import setup_style, save_figure, COLORS


# ---
#  LOAD ALL STATION RESULTS
# ---

def load_all_results():
    """Load Stage 3 results JSON from all stations."""
    model_dir = get_path('outputs_models')
    files = sorted(glob.glob(str(model_dir / 'station_*_results.json')))

    if not files:
        raise FileNotFoundError(
            f"No station_*_results.json in {model_dir}. Run Stage 3 first.")

    all_results = {}
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        # Extract station index from filename
        import re
        match = re.search(r'station_(\d+)', os.path.basename(fp))
        if match:
            sidx = int(match.group(1))
            all_results[sidx] = data
            print(f"  Loaded Station {sidx}: {len(data)} horizons")

    return all_results


# ---
#  BUILD SUMMARY TABLE
# ---

def build_summary_table(all_results):
    """Build a flat DataFrame of all station-horizon results."""
    rows = []
    for sidx, horizons in all_results.items():
        for hkey, data in horizons.items():
            h = int(hkey[1:])  # H1 -> 1, H96 -> 96
            row = {
                'station': sidx,
                'horizon': h,
                'horizon_min': h * 15,
                'horizon_label': hkey,
                'fame_r2': data.get('fame_r2'),
                'unified_r2': data.get('unified_r2'),
                'persist_r2': data.get('persist_r2'),
                'improvement_pp': data.get('improvement_pp'),
                'fss': data.get('fss'),
            }
            # Add component details if available
            if 'components' in data and data['components']:
                for comp, cdata in data['components'].items():
                    row[f'comp_{comp}_r2'] = cdata.get('r2')
                    row[f'comp_{comp}_model'] = cdata.get('model_type')
            if 'meta_weights' in data and data['meta_weights']:
                for comp, w in data['meta_weights'].items():
                    row[f'weight_{comp}'] = w
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(['station', 'horizon']).reset_index(drop=True)
    return df


# ---
#  STATISTICAL TESTS
# ---

def run_statistical_tests(df):
    """Run Wilcoxon signed-rank test and paired t-test."""
    from scipy import stats

    results = {}

    # Filter valid rows
    valid = df.dropna(subset=['fame_r2', 'unified_r2'])

    # Overall FAME vs Unified
    fame_vals = valid['fame_r2'].values
    unified_vals = valid['unified_r2'].values
    diff = fame_vals - unified_vals

    # Wilcoxon signed-rank test
    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(diff, alternative='greater')
    except ValueError:
        wilcoxon_stat, wilcoxon_p = np.nan, np.nan

    # Paired t-test
    ttest_stat, ttest_p = stats.ttest_rel(fame_vals, unified_vals, alternative='greater')

    results['overall'] = {
        'n_pairs': len(diff),
        'mean_improvement_pp': float(np.mean(diff) * 100),
        'median_improvement_pp': float(np.median(diff) * 100),
        'std_improvement_pp': float(np.std(diff) * 100),
        'fame_wins': int(np.sum(diff > 0)),
        'unified_wins': int(np.sum(diff < 0)),
        'ties': int(np.sum(diff == 0)),
        'win_rate': float(np.mean(diff > 0)),
        'wilcoxon_stat': float(wilcoxon_stat) if not np.isnan(wilcoxon_stat) else None,
        'wilcoxon_p': float(wilcoxon_p) if not np.isnan(wilcoxon_p) else None,
        'ttest_stat': float(ttest_stat),
        'ttest_p': float(ttest_p),
    }

    # Per-horizon tests
    for h in sorted(valid['horizon'].unique()):
        hdf = valid[valid['horizon'] == h]
        f_vals = hdf['fame_r2'].values
        u_vals = hdf['unified_r2'].values
        d = f_vals - u_vals

        try:
            ws, wp = stats.wilcoxon(d, alternative='greater')
        except ValueError:
            ws, wp = np.nan, np.nan

        ts, tp = stats.ttest_rel(f_vals, u_vals, alternative='greater')

        results[f'H{h}'] = {
            'n_stations': len(d),
            'mean_fame_r2': float(np.mean(f_vals)),
            'mean_unified_r2': float(np.mean(u_vals)),
            'mean_improvement_pp': float(np.mean(d) * 100),
            'fame_wins': int(np.sum(d > 0)),
            'wilcoxon_p': float(wp) if not np.isnan(wp) else None,
            'ttest_p': float(tp),
        }

    return results


# ---
#  DIAGNOSTIC PLOTS
# ---

def plot_cross_station_heatmap(df):
    """Heatmap: stations x horizons, color = improvement over Unified."""
    setup_style()

    stations = sorted(df['station'].unique())
    horizons = sorted(df['horizon'].unique())
    h_labels = [f'H{h}\n({h*15}m)' for h in horizons]

    # Build matrix
    matrix = np.full((len(stations), len(horizons)), np.nan)
    for i, s in enumerate(stations):
        for j, h in enumerate(horizons):
            row = df[(df['station'] == s) & (df['horizon'] == h)]
            if len(row) > 0 and row.iloc[0]['improvement_pp'] is not None:
                matrix[i, j] = row.iloc[0]['improvement_pp']

    fig, ax = plt.subplots(figsize=(12, 7))

    # Custom colormap: red for negative, white for zero, green for positive
    from matplotlib.colors import TwoSlopeNorm
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
    # Cap at reasonable range for readability (exclude Station 3 outliers)
    vmax_cap = min(vmax, 15)
    matrix_clipped = np.clip(matrix, -vmax_cap, vmax_cap)

    norm = TwoSlopeNorm(vmin=-vmax_cap, vcenter=0, vmax=vmax_cap)
    im = ax.imshow(matrix_clipped, cmap='RdYlGn', norm=norm, aspect='auto')

    # Labels
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels(h_labels, fontsize=10)
    ax.set_yticks(range(len(stations)))
    ax.set_yticklabels([f'Station {s}' for s in stations], fontsize=10)
    ax.set_xlabel('Forecast Horizon', fontsize=12)
    ax.set_title('FAME improvement over Unified XGBoost (pp)',
                 fontsize=13, fontweight='bold', color=COLORS['primary'])

    # Annotate cells
    for i in range(len(stations)):
        for j in range(len(horizons)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > vmax_cap * 0.6 else 'black'
                text = f'{val:+.1f}' if abs(val) < 100 else f'{val:+.0f}'
                ax.text(j, i, text, ha='center', va='center',
                        fontsize=9, fontweight='bold', color=color)

    plt.colorbar(im, ax=ax, label='Improvement (pp)', shrink=0.8)
    plt.tight_layout()
    save_figure(fig, 'cross_station_heatmap', stage='stage4')


def plot_horizon_curves(df):
    """R^2 vs horizon curves: FAME, Unified, Persistence (mean across stations)."""
    setup_style()

    # Exclude Station 3 (outlier) for clean mean curves
    df_clean = df[df['station'] != 3].copy()

    horizons = sorted(df_clean['horizon'].unique())
    h_mins = [h * 15 for h in horizons]

    fame_mean = [df_clean[df_clean['horizon'] == h]['fame_r2'].mean() for h in horizons]
    fame_std = [df_clean[df_clean['horizon'] == h]['fame_r2'].std() for h in horizons]
    unified_mean = [df_clean[df_clean['horizon'] == h]['unified_r2'].mean() for h in horizons]
    persist_mean = [df_clean[df_clean['horizon'] == h]['persist_r2'].mean() for h in horizons]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: R^2 curves
    ax = axes[0]
    ax.plot(h_mins, fame_mean, 'o-', color=COLORS['accent1'], lw=2.5, ms=9,
            label='FAME', zorder=5)
    ax.fill_between(h_mins,
                     [m - s for m, s in zip(fame_mean, fame_std)],
                     [m + s for m, s in zip(fame_mean, fame_std)],
                     alpha=0.15, color=COLORS['accent1'])
    ax.plot(h_mins, unified_mean, 's--', color=COLORS['accent3'], lw=1.5, ms=7,
            label='Unified XGBoost')
    ax.plot(h_mins, persist_mean, '^:', color=COLORS['neutral'], lw=1, ms=5,
            label='Persistence')
    ax.set_xlabel('Forecast Horizon (minutes)', fontsize=11)
    ax.set_ylabel('R^2', fontsize=11)
    ax.set_title('Mean R^2 across 7 stations (excl. Station 3)',
                 fontsize=12, fontweight='bold', color=COLORS['primary'])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.05)

    # Right: Improvement bars
    ax2 = axes[1]
    imp_mean = [(f - u) * 100 for f, u in zip(fame_mean, unified_mean)]
    colors = [COLORS['accent2'] if i > 0 else COLORS['accent1'] for i in imp_mean]
    bars = ax2.bar(range(len(horizons)), imp_mean, color=colors, alpha=0.85,
                    edgecolor='white', linewidth=0.5)
    ax2.axhline(y=0, color=COLORS['primary'], lw=0.8)
    ax2.set_xticks(range(len(horizons)))
    ax2.set_xticklabels([f'{m}m' for m in h_mins], fontsize=9)
    ax2.set_xlabel('Forecast Horizon', fontsize=11)
    ax2.set_ylabel('FAME - Unified (pp)', fontsize=11)
    ax2.set_title('Mean improvement per horizon',
                 fontsize=12, fontweight='bold', color=COLORS['primary'])
    ax2.grid(True, alpha=0.3, axis='y')

    # Annotate bars
    for bar, val in zip(bars, imp_mean):
        y = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, y + 0.05,
                 f'{val:+.2f}', ha='center', va='bottom', fontsize=8,
                 fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'horizon_curves_mean', stage='stage4')


def plot_per_station_summary(df):
    """Per-station radar/bar showing FAME advantage pattern."""
    setup_style()

    stations = sorted(df['station'].unique())
    n_stations = len(stations)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('Per-Station FAME vs Unified XGBoost',
                 fontsize=14, fontweight='bold', color=COLORS['primary'])

    for idx, sidx in enumerate(stations):
        ax = axes[idx // 4, idx % 4]
        sdf = df[df['station'] == sidx].sort_values('horizon')

        horizons = sdf['horizon_min'].values
        fame = sdf['fame_r2'].values
        unified = sdf['unified_r2'].values

        x = np.arange(len(horizons))
        width = 0.35
        ax.bar(x - width / 2, fame, width, color=COLORS['accent1'],
               alpha=0.85, label='FAME')
        ax.bar(x + width / 2, unified, width, color=COLORS['accent3'],
               alpha=0.85, label='Unified')

        ax.set_xticks(x)
        ax.set_xticklabels([f'{h}m' for h in horizons], fontsize=7)
        ax.set_title(f'Station {sidx}', fontsize=11, fontweight='bold')
        ax.set_ylim(min(0, min(fame.min(), unified.min()) - 0.05), 1.05)
        ax.grid(True, alpha=0.2, axis='y')

        if idx == 0:
            ax.legend(fontsize=7)

        # Win count
        wins = int(np.sum(fame > unified))
        ax.text(0.95, 0.05, f'FAME wins: {wins}/6',
                transform=ax.transAxes, fontsize=7, ha='right',
                color=COLORS['accent1'] if wins >= 4 else COLORS['accent3'])

    plt.tight_layout()
    save_figure(fig, 'per_station_bars', stage='stage4')


def plot_meta_weights(df):
    """Show how meta-learner weights shift across horizons."""
    setup_style()

    # Get weight columns
    weight_cols = [c for c in df.columns if c.startswith('weight_')]
    if not weight_cols:
        print("  No meta-weight data found, skipping weight plot")
        return

    # Use Station 1 as example (clean data)
    sdf = df[df['station'] == 1].sort_values('horizon')
    if len(sdf) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    horizons = sdf['horizon_min'].values
    band_names = [c.replace('weight_', '') for c in weight_cols]

    bottom = np.zeros(len(horizons))
    colors = [COLORS['accent1'], COLORS['accent2'], COLORS['accent4'], COLORS['neutral']]

    for i, (col, name) in enumerate(zip(weight_cols, band_names)):
        vals = sdf[col].fillna(0).values
        ax.bar(range(len(horizons)), vals, bottom=bottom,
               color=colors[i % len(colors)], alpha=0.8, label=name)
        bottom += vals

    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f'{h}m' for h in horizons], fontsize=9)
    ax.set_xlabel('Forecast Horizon', fontsize=11)
    ax.set_ylabel('Meta-learner Weight', fontsize=11)
    ax.set_title('Station 1: How component weights shift with horizon',
                 fontsize=12, fontweight='bold', color=COLORS['primary'])
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    save_figure(fig, 'meta_weights_shift', stage='stage4')


def plot_fss_heatmap(df):
    """Forecast Skill Score heatmap."""
    setup_style()

    stations = sorted(df['station'].unique())
    horizons = sorted(df['horizon'].unique())

    matrix = np.full((len(stations), len(horizons)), np.nan)
    for i, s in enumerate(stations):
        for j, h in enumerate(horizons):
            row = df[(df['station'] == s) & (df['horizon'] == h)]
            if len(row) > 0 and row.iloc[0]['fss'] is not None:
                matrix[i, j] = row.iloc[0]['fss']

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap='YlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f'H{h}\n({h*15}m)' for h in horizons], fontsize=9)
    ax.set_yticks(range(len(stations)))
    ax.set_yticklabels([f'Station {s}' for s in stations], fontsize=10)
    ax.set_title('Forecast Skill Score (FAME vs Persistence)',
                 fontsize=13, fontweight='bold', color=COLORS['primary'])

    for i in range(len(stations)):
        for j in range(len(horizons)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=9, fontweight='bold',
                        color='white' if val > 0.5 else 'black')

    plt.colorbar(im, ax=ax, label='FSS', shrink=0.8)
    plt.tight_layout()
    save_figure(fig, 'fss_heatmap', stage='stage4')


# ---
#  ENTRY POINT
# ---

def run():
    """Run the cross-station evaluation."""
    set_all_seeds()

    print(f"\n  Stage 4: Cross-Station Evaluation")
    print(f"  {'=' * 50}")

    # Load results
    all_results = load_all_results()
    print(f"  Loaded {len(all_results)} stations")

    # Build summary table
    df = build_summary_table(all_results)
    print(f"  Summary table: {len(df)} rows (station-horizon pairs)")

    # Save summary CSV
    reports_dir = get_path('outputs_reports')
    csv_path = reports_dir / 'stage4_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"  OK Summary CSV: {csv_path}")

    # -- Key statistics ---
    valid = df.dropna(subset=['improvement_pp'])
    total_pairs = len(valid)
    fame_wins = int((valid['improvement_pp'] > 0).sum())
    mean_imp = valid['improvement_pp'].mean()

    print(f"\n  -- KEY RESULTS --")
    print(f"  Total station-horizon pairs: {total_pairs}")
    print(f"  FAME wins: {fame_wins}/{total_pairs} ({fame_wins/total_pairs*100:.0f}%)")
    print(f"  Mean improvement: {mean_imp:+.4f} pp")

    # Exclude Station 3 for clean stats
    valid_clean = valid[valid['station'] != 3]
    mean_imp_clean = valid_clean['improvement_pp'].mean()
    fame_wins_clean = int((valid_clean['improvement_pp'] > 0).sum())
    total_clean = len(valid_clean)
    print(f"\n  Excluding Station 3 (noisy):")
    print(f"  FAME wins: {fame_wins_clean}/{total_clean} ({fame_wins_clean/total_clean*100:.0f}%)")
    print(f"  Mean improvement: {mean_imp_clean:+.4f} pp")

    # Per-horizon summary
    print(f"\n  Per-horizon (excl. Station 3):")
    print(f"  {'Horizon':>10} {'FAME':>10} {'Unified':>10} {'delta (pp)':>10} {'Wins':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for h in sorted(valid_clean['horizon'].unique()):
        hdf = valid_clean[valid_clean['horizon'] == h]
        fm = hdf['fame_r2'].mean()
        um = hdf['unified_r2'].mean()
        im = hdf['improvement_pp'].mean()
        w = int((hdf['improvement_pp'] > 0).sum())
        t = len(hdf)
        print(f"  H{h:>8} {fm:>10.6f} {um:>10.6f} {im:>+10.4f} {w:>4}/{t}")

    # -- Statistical tests ---
    print(f"\n  -- STATISTICAL TESTS --")
    stats = run_statistical_tests(valid_clean)

    overall = stats['overall']
    print(f"  Wilcoxon signed-rank (FAME > Unified):")
    print(f"    Statistic: {overall['wilcoxon_stat']}")
    print(f"    p-value: {overall['wilcoxon_p']}")
    sig = overall['wilcoxon_p'] < 0.05 if overall['wilcoxon_p'] else False
    print(f"    Significant at -=0.05: {'YES' if sig else 'NO'}")

    print(f"  Paired t-test:")
    print(f"    t-statistic: {overall['ttest_stat']:.4f}")
    print(f"    p-value: {overall['ttest_p']:.6f}")

    # -- Generate plots ---
    print(f"\n  -- GENERATING PLOTS --")

    try:
        plot_cross_station_heatmap(df)
        print(f"  OK Cross-station heatmap")
    except Exception as e:
        print(f"  ! Heatmap failed: {e}")

    try:
        plot_horizon_curves(df)
        print(f"  OK Horizon R^2 curves")
    except Exception as e:
        print(f"  ! Horizon curves failed: {e}")

    try:
        plot_per_station_summary(df)
        print(f"  OK Per-station bar charts")
    except Exception as e:
        print(f"  ! Per-station bars failed: {e}")

    try:
        plot_meta_weights(df)
        print(f"  OK Meta-weight evolution plot")
    except Exception as e:
        print(f"  ! Meta-weights failed: {e}")

    try:
        plot_fss_heatmap(df)
        print(f"  OK FSS heatmap")
    except Exception as e:
        print(f"  ! FSS heatmap failed: {e}")
    # -- Save all metrics ---
    eval_metrics = {
        'overall': overall,
        'per_horizon': {k: v for k, v in stats.items() if k != 'overall'},
        'total_pairs': total_pairs,
        'fame_wins': fame_wins,
        'win_rate': fame_wins / total_pairs if total_pairs > 0 else 0,
    }
    save_metrics(eval_metrics, 'stage4', get_path('outputs_artifacts'))

    print(f"\n  - Stage 4 complete")
    return eval_metrics


if __name__ == "__main__":
    from utils.config import manifest as cm
    m = cm("stage4_evaluation")
    try:
        result = run()
    except Exception as e:
        m.log_param("error", str(e))
        raise
    finally:
        print(f"  OK Manifest: {m.save()}")
