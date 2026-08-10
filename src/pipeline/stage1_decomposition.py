"""
Stage 1: signal decomposition.

Decomposes the irradiance series into frequency components with wavelet, EMD
and VMD, compares them against a no-decomposition baseline on validation R^2,
and writes the winning method's components into the dataframe.

Scope of the transform:
  The decomposition runs over the full series, train through test. No
  parameters are fitted from the data, so no model state crosses the split.
  The transforms are not causal filters, however: wavelet reconstruction at
  time t uses samples on both sides of t, so a reconstructed band carries
  information from beyond t. Band features therefore support a filtering
  interpretation of the results rather than a strictly causal forecasting one.
  Use decompose_causal() for a strictly causal alternative.

Input:  data/processed/station_XX_prepared.csv
Output: data/processed/station_XX_decomposed.csv
        outputs/artifacts/stage1_metrics.json
        outputs/plots/stage1/*.png

    python -m pipeline.stage1_decomposition              # all stations
    python -m pipeline.stage1_decomposition --station 5  # single station
"""
import os, sys, glob, time, argparse, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# -- Project imports ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import cfg, get_path, set_all_seeds, RunManifest
from utils.schema import check_temporal_leakage, SchemaError
from utils.metrics import compute_all, save_metrics, bootstrap_ci
from utils.plotting import setup_style, save_figure, COLORS, COMPONENT_COLORS


# ---
#  DECOMPOSITION METHODS
# ---

def decompose_wavelet(signal, family='db4', level=3):
    """
    Wavelet decomposition of signal into approximation + detail components.
    Returns dict of {component_name: array}.
    """
    import pywt
    coeffs = pywt.wavedec(signal, family, level=level)
    components = {}

    for i, c in enumerate(coeffs):
        # Zero out all coefficients except current
        zeroed = [np.zeros_like(cc) for cc in coeffs]
        zeroed[i] = c
        rec = pywt.waverec(zeroed, family)

        # Handle length mismatch from wavelet reconstruction
        if len(rec) > len(signal):
            rec = rec[:len(signal)]
        elif len(rec) < len(signal):
            rec = np.pad(rec, (0, len(signal) - len(rec)), mode='edge')

        if i == 0:
            name = f'IRR_WAV_cA{level}'
        else:
            name = f'IRR_WAV_cD{level - i + 1}'
        components[name] = rec

    # Verify reconstruction
    recon = sum(components.values())
    recon_error = np.sqrt(np.mean((signal - recon) ** 2))

    return components, recon_error


def decompose_causal(signal, family='db4', level=3, window=512):
    """Strictly causal wavelet band reconstruction.

    Returns the same component names and shapes as decompose_wavelet(), but the
    value at index t is computed from signal[:t+1] only: the window ending at t
    is decomposed and the last reconstructed sample of each band is taken, so no
    sample ever sees its own future.

    A sliding window of fixed length keeps the cost at O(n*window) rather than
    the O(n^2) of decomposing an expanding prefix at every step. The default 512
    samples span about 5.3 days at 15-minute resolution, which covers the
    level-3 db4 support and the diurnal cycle.

    Reconstruction error is higher than the whole-series transform because the
    boundary at t is one-sided; that is inherent to causal filtering.
    """
    import pywt

    n = len(signal)
    n_bands = level + 1
    out = np.full((n, n_bands), np.nan, dtype=float)

    min_len = 2 ** (level + 1)
    span = max(window, 4 * 2 ** level)

    for t in range(n):
        segment = signal[max(0, t - span + 1):t + 1]
        if len(segment) < min_len:
            # Too little history for a level-`level` transform: carry the raw
            # value in the approximation band and leave the details at zero.
            out[t, 0] = signal[t]
            out[t, 1:] = 0.0
            continue

        coeffs = pywt.wavedec(segment, family, level=level)
        for i in range(n_bands):
            zeroed = [np.zeros_like(c) for c in coeffs]
            zeroed[i] = coeffs[i]
            rec = pywt.waverec(zeroed, family)
            out[t, i] = rec[:len(segment)][-1]

    components = {}
    for i in range(n_bands):
        name = (f'IRR_WAV_cA{level}' if i == 0
                else f'IRR_WAV_cD{level - i + 1}')
        components[name] = out[:, i]

    recon = sum(components.values())
    recon_error = float(np.sqrt(np.nanmean((signal - recon) ** 2)))
    return components, recon_error


def decompose_emd(signal, max_imfs=15):
    """
    Empirical Mode Decomposition into Intrinsic Mode Functions.
    Returns dict of {component_name: array}.
    """
    from PyEMD import EMD

    emd = EMD()
    emd.MAX_ITERATION = 500
    imfs = emd.emd(signal)

    components = {}
    for i in range(min(imfs.shape[0], max_imfs)):
        components[f'IRR_EMD_IMF{i + 1}'] = imfs[i]

    # Verify reconstruction
    recon = sum(components.values())
    recon_error = np.sqrt(np.mean((signal - recon) ** 2))

    return components, recon_error


def decompose_vmd(signal, K=5, alpha=2000):
    """
    Variational Mode Decomposition into K modes.
    Returns dict of {component_name: array}.
    """
    from vmdpy import VMD

    # VMD parameters
    tau = 0       # noise tolerance
    DC = 0        # no DC part imposed
    init = 1      # uniform initialization
    tol = 1e-7    # convergence tolerance

    modes, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)

    components = {}
    for i in range(modes.shape[0]):
        components[f'IRR_VMD_M{i + 1}'] = modes[i]

    # Verify reconstruction
    recon = sum(components.values())
    recon_error = np.sqrt(np.mean((signal - recon) ** 2))

    return components, recon_error


# ---
#  EVALUATION HELPER
# ---

def evaluate_decomposition(df, base_features, extra_components, train_idx, test_idx):
    """
    Train XGBoost with base features + decomposition components,
    evaluate on test set. Returns metrics dict.
    """
    from xgboost import XGBRegressor

    target = 'TARGET_H1'  # Use 1-step ahead for decomposition comparison
    y_all = df[target].values

    # Build feature matrix
    feature_names = list(base_features)
    X = df[feature_names].values.copy()

    if extra_components:
        comp_arrays = np.column_stack(list(extra_components.values()))
        X = np.hstack([X, comp_arrays])
        feature_names += list(extra_components.keys())

    # Remove rows where target is NaN
    valid = ~np.isnan(y_all)
    train_mask = valid & np.isin(np.arange(len(df)), train_idx)
    test_mask = valid & np.isin(np.arange(len(df)), test_idx)

    X_tr = X[train_mask]
    y_tr = y_all[train_mask]
    X_te = X[test_mask]
    y_te = y_all[test_mask]

    if len(X_tr) == 0 or len(X_te) == 0:
        return {'r2': -999, 'mae': 999, 'rmse': 999}, None

    model = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=cfg['seeds']['global'],
        n_jobs=-1, verbosity=0
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    metrics = compute_all(y_te, pred)
    metrics['n_features'] = len(feature_names)

    # Feature importance
    fi = dict(zip(feature_names, model.feature_importances_))
    metrics['top_features'] = dict(sorted(fi.items(), key=lambda x: -x[1])[:5])

    return metrics, pred


# ---
#  PLOTTING
# ---

def plot_decomposition(signal, components, method_name, station_idx):
    """Plot original signal and its decomposition components."""
    setup_style()

    n_comp = len(components)
    fig, axes = plt.subplots(n_comp + 1, 1, figsize=(18, 2.5 * (n_comp + 1)), sharex=True)
    fig.suptitle(f'Station {station_idx}: {method_name} Decomposition of Irradiation',
                 fontsize=14, fontweight='bold', color=COLORS['primary'])

    # Show first 1000 points for clarity
    n_show = min(1000, len(signal))

    # Original
    axes[0].plot(signal[:n_show], color=COLORS['primary'], linewidth=0.6, alpha=0.9)
    axes[0].fill_between(range(n_show), signal[:n_show], alpha=0.08, color=COLORS['primary'])
    axes[0].set_ylabel('Original', fontsize=9)

    # Components
    comp_colors = [COLORS['accent1'], COLORS['accent2'], COLORS['accent4'],
                   COLORS['accent3'], COLORS['accent5'], COLORS['accent6']]
    for j, (name, comp) in enumerate(components.items()):
        color = comp_colors[j % len(comp_colors)]
        axes[j + 1].plot(comp[:n_show], color=color, linewidth=0.6, alpha=0.9)
        axes[j + 1].fill_between(range(n_show), comp[:n_show], alpha=0.06, color=color)
        axes[j + 1].set_ylabel(name, fontsize=7)

    axes[-1].set_xlabel(f'Sample index (first {n_show})', fontsize=10)
    plt.tight_layout()
    save_figure(fig, f'station_{station_idx:02d}_{method_name.lower()}_decomposition', stage='stage1')


def plot_comparison(results, station_idx):
    """Plot R^2 comparison across decomposition methods."""
    setup_style()

    methods = sorted(results.keys(), key=lambda k: -results[k]['r2'])
    r2_vals = [results[m]['r2'] for m in methods]
    n_feats = [results[m]['n_features'] for m in methods]
    colors = [COLORS['accent1'] if m == methods[0] else COLORS['accent3'] for m in methods]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Lollipop chart
    y_pos = range(len(methods))
    ax.hlines(y_pos, xmin=min(r2_vals) - 0.01, xmax=r2_vals, color=colors, linewidth=3, alpha=0.7)
    ax.scatter(r2_vals, y_pos, s=120, c=colors, zorder=5, edgecolors='white', linewidth=1.5)

    for i, (r2, nf) in enumerate(zip(r2_vals, n_feats)):
        ax.text(r2 + 0.001, i, f'{r2:.6f} ({nf} feat)', va='center', fontsize=9,
                fontweight='bold', color=COLORS['primary'])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_xlabel('R^2 Score (1-step ahead forecast)', fontsize=11)
    ax.set_title(f'Station {station_idx}: Decomposition Method Comparison',
                 fontsize=13, fontweight='bold', color=COLORS['primary'])

    # Mark baseline
    baseline_r2 = results.get('No Decomposition', {}).get('r2', 0)
    if baseline_r2 > 0:
        ax.axvline(x=baseline_r2, color=COLORS['neutral'], linestyle='--',
                   linewidth=1, alpha=0.7, label='Baseline')
        ax.legend(loc='lower right')

    plt.tight_layout()
    save_figure(fig, f'station_{station_idx:02d}_decomposition_comparison', stage='stage1')


# ---
#  PROCESS ONE STATION
# ---

def process_station(filepath, manifest=None):
    """Run full decomposition comparison for one station."""
    fname = os.path.basename(filepath)
    # Extract station index from filename
    import re
    match = re.search(r'station_(\d+)', fname)
    if not match:
        print(f"  ! Cannot parse station index from {fname}, skipping")
        return None
    station_idx = int(match.group(1))

    print(f"\n{'-' * 70}")
    print(f"  Station {station_idx}: {fname}")
    print(f"{'-' * 70}")

    # Load
    df = pd.read_csv(filepath, parse_dates=['DATE_TIME'])
    signal = df['IRRADIATION'].values
    print(f"  Loaded: {len(df):,} rows")

    if manifest:
        manifest.log_input(fname, Path(filepath))

    # Indices for train/test (skip val for decomposition comparison)
        train_idx = np.where(df['SPLIT'] == 'train')[0].astype(int)
        val_idx   = np.where(df['SPLIT'] == 'val')[0].astype(int)
        test_idx  = np.where(df['SPLIT'] == 'test')[0].astype(int)   # keep for reference, not used in scoring

    # Check TARGET_H1 exists
    if 'TARGET_H1' not in df.columns:
        print(f"  FAIL TARGET_H1 not found  -  run Stage 0 first")
        return None

    # Base features (no decomposition)
    base_features = ['IRRADIATION', 'TEMPERATURE', 'HOUR', 'DAY', 'MONTH', 'DOY']
    # Add extra weather cols if available
    for col in ['REL_HUMIDITY', 'ATMOSPHERE', 'GHI', 'DNI']:
        if col in df.columns:
            base_features.append(col)

    results = {}

    # -- Method 0: No Decomposition (baseline) ---
    print(f"\n  [0] No Decomposition (baseline)...")
    t0 = time.time()
    m_base, _ = evaluate_decomposition(df, base_features, {}, train_idx, val_idx)
    m_base['time_s'] = time.time() - t0
    results['No Decomposition'] = m_base
    print(f"      R^2={m_base['r2']:.6f} ({len(base_features)} features, {m_base['time_s']:.1f}s)")

    # -- Method 1: Wavelet ---
    stage1_cfg = cfg.get('stage1', {})
    wav_cfg = stage1_cfg.get('wavelet', {})
    wav_family = wav_cfg.get('family', 'db4')
    wav_levels = wav_cfg.get('levels', [3, 4])

    best_wav = None
    for level in wav_levels:
        label = f'Wavelet ({wav_family}, L{level})'
        print(f"\n  [1] {label}...")
        t0 = time.time()
        try:
            components, recon_err = decompose_wavelet(signal, wav_family, level)
            m_wav, _ = evaluate_decomposition(df, base_features, components, train_idx, val_idx)
            m_wav['time_s'] = time.time() - t0
            m_wav['recon_error'] = float(recon_err)
            m_wav['n_components'] = len(components)
            results[label] = m_wav
            print(f"      R^2={m_wav['r2']:.6f} ({m_wav['n_features']} feat, "
                  f"recon_err={recon_err:.8f}, {m_wav['time_s']:.1f}s)")

            if best_wav is None or m_wav['r2'] > best_wav[1]['r2']:
                best_wav = (label, m_wav, components)

            # Plot decomposition for best level
            plot_decomposition(signal, components, label, station_idx)

        except Exception as e:
            print(f"      FAIL Failed: {e}")

    # -- Method 2: EMD ---
    emd_cfg = stage1_cfg.get('emd', {})
    max_imfs = emd_cfg.get('max_imfs', 15)
    print(f"\n  [2] EMD (max {max_imfs} IMFs)...")
    t0 = time.time()
    try:
        components_emd, recon_err_emd = decompose_emd(signal, max_imfs)
        m_emd, _ = evaluate_decomposition(df, base_features, components_emd, train_idx, val_idx)
        m_emd['time_s'] = time.time() - t0
        m_emd['recon_error'] = float(recon_err_emd)
        m_emd['n_components'] = len(components_emd)
        results['EMD'] = m_emd
        print(f"      R^2={m_emd['r2']:.6f} ({m_emd['n_features']} feat, "
              f"{len(components_emd)} IMFs, recon_err={recon_err_emd:.8f}, {m_emd['time_s']:.1f}s)")

        # Plot only first 6 components (EMD can produce many)
        plot_components = dict(list(components_emd.items())[:6])
        plot_decomposition(signal, plot_components, 'EMD', station_idx)

    except Exception as e:
        print(f"      FAIL EMD failed: {e}")
        components_emd = None

    # -- Method 3: VMD ---
    vmd_cfg = stage1_cfg.get('vmd', {})
    K_range = vmd_cfg.get('K_range', [3, 8])
    alpha = vmd_cfg.get('alpha', 2000)

    best_vmd = None
    for K in range(K_range[0], K_range[1] + 1, 2):  # Test K=3,5,7
        label = f'VMD (K={K})'
        print(f"\n  [3] {label}...")
        t0 = time.time()
        try:
            components_vmd, recon_err_vmd = decompose_vmd(signal, K=K, alpha=alpha)
            m_vmd, _ = evaluate_decomposition(df, base_features, components_vmd, train_idx, val_idx)
            m_vmd['time_s'] = time.time() - t0
            m_vmd['recon_error'] = float(recon_err_vmd)
            m_vmd['n_components'] = len(components_vmd)
            results[label] = m_vmd
            print(f"      R^2={m_vmd['r2']:.6f} ({m_vmd['n_features']} feat, "
                  f"recon_err={recon_err_vmd:.8f}, {m_vmd['time_s']:.1f}s)")

            if best_vmd is None or m_vmd['r2'] > best_vmd[1]['r2']:
                best_vmd = (label, m_vmd, components_vmd)

        except Exception as e:
            print(f"      FAIL VMD K={K} failed: {e}")

    # Plot best VMD
    if best_vmd:
        plot_decomposition(signal, best_vmd[2], best_vmd[0], station_idx)

    # -- SELECT WINNER ---
    winner_name = max(results, key=lambda k: results[k]['r2'])
    winner_metrics = results[winner_name]
    baseline_r2 = results['No Decomposition']['r2']
    delta_pp = (winner_metrics['r2'] - baseline_r2) * 100

    print(f"\n  {'-' * 50}")
    print(f"  Results ranking:")
    for rank, (name, m) in enumerate(sorted(results.items(), key=lambda x: -x[1]['r2']), 1):
        marker = '-' if name == winner_name else '  '
        delta = (m['r2'] - baseline_r2) * 100
        print(f"    {marker} {rank}. {name}: R^2={m['r2']:.6f} (delta ={delta:+.4f} pp, {m['n_features']} feat)")

    print(f"\n  Winner: {winner_name} (R^2={winner_metrics['r2']:.6f}, delta ={delta_pp:+.4f} pp)")

    # -- ADD WINNER COMPONENTS TO DATAFRAME ---
    # Determine which components to inject
    if winner_name == 'No Decomposition':
        winner_components = {}
    elif 'Wavelet' in winner_name and best_wav:
        winner_components = best_wav[2]
    elif winner_name == 'EMD' and components_emd:
        winner_components = components_emd
    elif 'VMD' in winner_name and best_vmd:
        winner_components = best_vmd[2]
    else:
        winner_components = {}

    for col_name, col_vals in winner_components.items():
        df[col_name] = col_vals

    # Plot comparison
    plot_comparison(results, station_idx)

    # -- SAVE ---
    proc_dir = get_path('data_processed')
    out_name = f'station_{station_idx:02d}_decomposed.csv'
    out_path = proc_dir / out_name
    df.to_csv(out_path, index=False)
    print(f"\n  OK Saved: {out_name} ({len(df):,} rows x {len(df.columns)} cols)")

    if manifest:
        manifest.log_output(out_name, out_path)
        manifest.log_metric(f'station_{station_idx}_winner', winner_name)
        manifest.log_metric(f'station_{station_idx}_r2', winner_metrics['r2'])
        manifest.log_metric(f'station_{station_idx}_baseline_r2', baseline_r2)
        manifest.log_metric(f'station_{station_idx}_delta_pp', delta_pp)

    return {
        'station': station_idx,
        'winner': winner_name,
        'r2': winner_metrics['r2'],
        'baseline_r2': baseline_r2,
        'delta_pp': delta_pp,
        'n_components': len(winner_components),
        'all_results': {k: {'r2': v['r2'], 'n_features': v['n_features']}
                        for k, v in results.items()},
    }


# ---
#  MAIN ORCHESTRATION
# ---

def run(station_idx=None, manifest=None):
    """Process all (or one) stations."""
    seed = set_all_seeds()

    proc_dir = get_path('data_processed')
    files = sorted(glob.glob(str(proc_dir / 'station_*_prepared.csv')))

    if not files:
        raise FileNotFoundError(f"No station_*_prepared.csv files in {proc_dir}. Run Stage 0 first.")

    if station_idx is not None:
        files = [f for f in files if f'station_{station_idx:02d}_' in f]
        if not files:
            raise FileNotFoundError(f"No prepared file for station {station_idx}")

    print(f"\n  Stage 1: Signal Decomposition")
    print(f"  Found {len(files)} station file(s)")
    print(f"  Methods: Wavelet, EMD, VMD, No Decomposition")

    all_summaries = []

    for filepath in files:
        summary = process_station(filepath, manifest)
        if summary:
            all_summaries.append(summary)

    # -- CROSS-STATION SUMMARY ---
    if all_summaries:
        print(f"\n{'-' * 70}")
        print(f"  STAGE 1 SUMMARY: Decomposition Winners")
        print(f"{'-' * 70}")
        print(f"  {'Station':>8} {'Winner':<25} {'R^2':>10} {'Baseline':>10} {'Gain (pp)':>10}")
        print(f"  {'-' * 8} {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10}")
        for s in all_summaries:
            print(f"  {s['station']:>8} {s['winner']:<25} {s['r2']:>10.6f} "
                  f"{s['baseline_r2']:>10.6f} {s['delta_pp']:>+10.4f}")

        # Save metrics
        save_metrics(
            {'stations': all_summaries},
            'stage1', get_path('outputs_artifacts')
        )

        # Winner frequency table
        from collections import Counter
        winner_counts = Counter(s['winner'] for s in all_summaries)
        print(f"\n  Winner frequency: {dict(winner_counts)}")
        mean_gain = np.mean([s['delta_pp'] for s in all_summaries])
        print(f"  Mean gain over baseline: {mean_gain:+.4f} pp")

    print(f"\n  - Stage 1 complete")
    return all_summaries


# ---
#  STANDALONE EXECUTION
# ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Signal Decomposition")
    parser.add_argument("--station", type=int, default=None,
                       help="Process specific station (1-8)")
    args = parser.parse_args()

    from utils.config import manifest as create_manifest
    m = create_manifest("stage1_decomposition")

    try:
        run(station_idx=args.station, manifest=m)
    except Exception as e:
        m.log_param("error", str(e))
        raise
    finally:
        manifest_path = m.save()
        print(f"  OK Manifest: {manifest_path}")
