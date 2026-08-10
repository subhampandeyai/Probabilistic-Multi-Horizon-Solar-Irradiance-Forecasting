"""
Stage 2: feature construction.

Builds one shared feature matrix from the raw signal, the decomposition
components and the meteorological covariates: lagged values, rolling mean and
standard deviation, first and second differences, sinusoidal calendar
encodings, and per-band derivatives of each component.

CORRECTNESS REQUIREMENT: every feature is backward-looking. Rolling windows are
trailing and lags use shift(+n); a forward shift here would leak the target into
its own predictors and silently inflate every downstream score.

Input:  data/processed/station_XX_decomposed.csv
Output: data/processed/station_XX_features.csv
        outputs/artifacts/stage2_metrics.json

    python -m src.pipeline.stage2_feature_engineering
"""
import os, sys, glob, time, argparse, warnings, re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import cfg, get_path, set_all_seeds, RunManifest
from utils.schema import check_temporal_leakage, SchemaError
from utils.metrics import compute_all, save_metrics
from utils.plotting import setup_style, save_figure, COLORS


def detect_decomposition_cols(df):
    cols = list(df.columns)
    decomp = {'trend': [], 'daily': [], 'hourly': [], 'noise': [], 'all_decomp': []}
    wav_cols = sorted([c for c in cols if c.startswith('IRR_WAV_')])
    emd_cols = sorted([c for c in cols if c.startswith('IRR_EMD_')])
    vmd_cols = sorted([c for c in cols if c.startswith('IRR_VMD_')])
    if wav_cols:
        decomp['all_decomp'] = wav_cols
        ca_cols = [c for c in wav_cols if '_cA' in c]
        cd_cols = sorted([c for c in wav_cols if '_cD' in c],
                        key=lambda x: int(re.search(r'cD(\d+)', x).group(1)), reverse=True)
        decomp['trend'] = ca_cols
        if len(cd_cols) >= 3:
            decomp['daily'] = [cd_cols[0]]
            decomp['hourly'] = [cd_cols[1]]
            decomp['noise'] = cd_cols[2:]
        elif len(cd_cols) == 2:
            decomp['daily'] = [cd_cols[0]]
            decomp['noise'] = [cd_cols[1]]
        elif len(cd_cols) == 1:
            decomp['noise'] = cd_cols
    elif emd_cols:
        decomp['all_decomp'] = emd_cols
        n = len(emd_cols)
        decomp['trend'] = emd_cols[-1:]
        decomp['daily'] = emd_cols[n//2:n-1]
        decomp['hourly'] = emd_cols[1:n//2]
        decomp['noise'] = emd_cols[:1]
    elif vmd_cols:
        decomp['all_decomp'] = vmd_cols
        n = len(vmd_cols)
        decomp['trend'] = vmd_cols[:1]
        decomp['daily'] = vmd_cols[1:max(2, n//3)]
        decomp['hourly'] = vmd_cols[max(2, n//3):max(3, 2*n//3)]
        decomp['noise'] = vmd_cols[max(3, 2*n//3):]
    return decomp


def add_rolling_features(df, col, windows_steps, prefix=None):
    if prefix is None: prefix = col
    new_cols = {}
    for ws in windows_steps:
        label = f'{ws*15}min' if ws <= 4 else f'{ws//4}h'
        new_cols[f'{prefix}_rmean_{label}'] = df[col].rolling(ws, min_periods=1).mean()
        new_cols[f'{prefix}_rstd_{label}'] = df[col].rolling(ws, min_periods=1).std().fillna(0)
        new_cols[f'{prefix}_rmin_{label}'] = df[col].rolling(ws, min_periods=1).min()
        new_cols[f'{prefix}_rmax_{label}'] = df[col].rolling(ws, min_periods=1).max()
    return new_cols


def add_lag_features(df, col, lags, prefix=None):
    if prefix is None: prefix = col
    new_cols = {}
    for lag in lags:
        new_cols[f'{prefix}_lag{lag}'] = df[col].shift(lag)
    return new_cols


def add_delta_features(df, col, prefix=None):
    if prefix is None: prefix = col
    delta = df[col].diff()
    return {f'{prefix}_delta': delta, f'{prefix}_accel': delta.diff()}


def add_solar_geometry(df, lat_deg=35.0):
    LAT = np.radians(lat_deg)
    hours = df['HOUR'].values; doy = df['DOY'].values
    B = (2 * np.pi / 365) * (doy - 81)
    decl_rad = np.radians(23.45 * np.sin(B))
    ha_rad = np.radians((hours - 12) * 15)
    sin_elev = np.clip(np.sin(LAT)*np.sin(decl_rad) + np.cos(LAT)*np.cos(decl_rad)*np.cos(ha_rad), 0, 1)
    ext_rad = np.maximum(1361 * sin_elev, 0.001)
    clear_sky = np.clip(df['IRRADIATION'].values / (ext_rad / 1000), 0, 2)
    zenith = np.clip(90 - np.degrees(np.arcsin(sin_elev)), 0, 89)
    air_mass = np.clip(1/(np.cos(np.radians(zenith)) + 0.50572*(96.07995-zenith)**(-1.6364)), 1, 40)
    cos_ha_ss = np.clip(-np.tan(LAT)*np.tan(decl_rad), -1, 1)
    dl = (2/15)*np.degrees(np.arccos(cos_ha_ss))
    day_frac = np.clip((hours-(12-dl/2))/np.maximum(dl, 0.1), 0, 1)
    return {'EXTRATERRESTRIAL_RAD': ext_rad, 'CLEAR_SKY_INDEX': clear_sky,
            'AIR_MASS': air_mass, 'DAY_FRACTION': day_frac}


def add_polynomial_features(df):
    irr = df['IRRADIATION'].clip(lower=0).values
    new_cols = {'IRR_log': np.log1p(irr * 1000), 'IRR_sq': irr ** 2}
    if 'TEMPERATURE' in df.columns:
        new_cols['TEMP_sq'] = df['TEMPERATURE'].values ** 2
    return new_cols


def add_cyclical_encoding(df):
    return {
        'HALFDAY_sin': np.sin(2*np.pi*df['HOUR'].values/12),
        'HALFDAY_cos': np.cos(2*np.pi*df['HOUR'].values/12),
        'DAY_sin': np.sin(2*np.pi*df['DAY'].values/30),
        'DAY_cos': np.cos(2*np.pi*df['DAY'].values/30),
        'YEAR_sin': np.sin(2*np.pi*df['DOY'].values/365),
        'YEAR_cos': np.cos(2*np.pi*df['DOY'].values/365),
    }


def add_interaction_features(df):
    new_cols = {}
    if 'CLEAR_SKY_INDEX' in df.columns and 'TEMPERATURE' in df.columns:
        new_cols['CSI_x_TEMP'] = df['CLEAR_SKY_INDEX'].values * df['TEMPERATURE'].values
    if 'IRRADIATION' in df.columns and 'TEMPERATURE' in df.columns:
        new_cols['IRR_x_TEMP'] = df['IRRADIATION'].values * df['TEMPERATURE'].values
    return new_cols


def build_component_features(df, decomp_info, stage2_cfg):
    rolling_windows = stage2_cfg.get('rolling_windows_steps', [4, 12, 48])
    lag_steps = stage2_cfg.get('lag_steps', [1, 2, 3, 4, 8, 12])
    lat_deg = cfg['dataset']['chinese_stations'].get('latitude_deg', 35.0)
    all_new = {}; feature_groups = {}

    # A: Irradiation features
    grp = {}
    grp.update(add_rolling_features(df, 'IRRADIATION', rolling_windows, 'IRR'))
    grp.update(add_lag_features(df, 'IRRADIATION', lag_steps[:4], 'IRR'))
    grp.update(add_delta_features(df, 'IRRADIATION', 'IRR'))
    grp.update(add_polynomial_features(df))
    all_new.update(grp); feature_groups['A_irradiation'] = list(grp.keys())

    # B: Solar geometry
    if stage2_cfg.get('solar_geometry', True):
        grp = add_solar_geometry(df, lat_deg)
        all_new.update(grp); feature_groups['B_solar_geometry'] = list(grp.keys())

    # C: Cyclical
    if stage2_cfg.get('cyclical_encoding', True):
        grp = add_cyclical_encoding(df)
        all_new.update(grp); feature_groups['C_cyclical'] = list(grp.keys())

    # D: Temperature
    grp = {}
    if 'TEMPERATURE' in df.columns:
        grp.update(add_rolling_features(df, 'TEMPERATURE', [4, 12], 'TEMP'))
        grp.update(add_lag_features(df, 'TEMPERATURE', [1, 2], 'TEMP'))
        grp.update(add_delta_features(df, 'TEMPERATURE', 'TEMP'))
    all_new.update(grp); feature_groups['D_temperature'] = list(grp.keys())

    # E: Interactions (need solar geometry cols in df first)
    temp_df = df.copy()
    for k, v in all_new.items():
        temp_df[k] = v if not isinstance(v, pd.Series) else v.values
    grp = add_interaction_features(temp_df)
    all_new.update(grp); feature_groups['E_interactions'] = list(grp.keys())

    # F: COMPONENT-SPECIFIC (NOVEL)
    grp = {}
    for col in decomp_info['trend']:
        grp.update(add_rolling_features(df, col, [12, 48], col))
        grp.update(add_lag_features(df, col, [4, 12], col))
    for col in decomp_info['daily']:
        grp.update(add_rolling_features(df, col, [4, 12], col))
        grp.update(add_lag_features(df, col, [1, 2, 4], col))
        grp.update(add_delta_features(df, col, col))
    for col in decomp_info['hourly']:
        grp.update(add_rolling_features(df, col, [4], col))
        grp.update(add_lag_features(df, col, [1, 2, 3], col))
        grp.update(add_delta_features(df, col, col))
    for col in decomp_info['noise']:
        grp.update(add_lag_features(df, col, [1], col))
    all_new.update(grp); feature_groups['F_component_specific'] = list(grp.keys())

    # G: Extra weather
    grp = {}
    for col in ['REL_HUMIDITY', 'ATMOSPHERE']:
        if col in df.columns:
            grp.update(add_lag_features(df, col, [1, 4], col))
    all_new.update(grp); feature_groups['G_extra_weather'] = list(grp.keys())

    return all_new, feature_groups


def evaluate_features(df, feature_cols, target_col, train_idx, test_idx):
    from xgboost import XGBRegressor
    y = df[target_col].values; X = df[feature_cols].values
    valid = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    train_mask = valid & np.isin(np.arange(len(df)), train_idx)
    test_mask = valid & np.isin(np.arange(len(df)), test_idx)
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]
    if len(X_tr) < 100 or len(X_te) < 50:
        return {'r2': -999, 'n_features': len(feature_cols)}, None
    model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=cfg['seeds']['global'], n_jobs=-1, verbosity=0)
    model.fit(X_tr, y_tr); pred = model.predict(X_te)
    metrics = compute_all(y_te, pred)
    metrics['n_features'] = len(feature_cols)
    metrics['n_train'] = int(train_mask.sum())
    metrics['n_test'] = int(test_mask.sum())
    fi = dict(zip(feature_cols, model.feature_importances_))
    metrics['top_features'] = dict(sorted(fi.items(), key=lambda x: -x[1])[:10])
    return metrics, model


def plot_feature_groups(group_results, station_idx):
    setup_style()
    groups = sorted(group_results.keys(), key=lambda k: -group_results[k]['r2'])
    r2_vals = [group_results[g]['r2'] for g in groups]
    n_feats = [group_results[g]['n_features'] for g in groups]
    colors = [COLORS['accent1'] if g == groups[0] else
              COLORS['accent2'] if 'component' in g.lower() else COLORS['accent3'] for g in groups]
    fig, ax = plt.subplots(figsize=(12, max(4, len(groups)*0.6)))
    y_pos = range(len(groups))
    ax.hlines(y_pos, xmin=min(r2_vals)-0.005, xmax=r2_vals, color=colors, linewidth=3, alpha=0.7)
    ax.scatter(r2_vals, y_pos, s=120, c=colors, zorder=5, edgecolors='white', linewidth=1.5)
    for i, (r2, nf) in enumerate(zip(r2_vals, n_feats)):
        ax.text(r2+0.001, i, f'{r2:.6f} ({nf} feat)', va='center', fontsize=9, fontweight='bold', color=COLORS['primary'])
    ax.set_yticks(y_pos); ax.set_yticklabels([g.replace('_', ' ') for g in groups], fontsize=9)
    ax.set_xlabel('R^2 Score (1-step ahead)', fontsize=11)
    ax.set_title(f'Station {station_idx}: Feature Group Comparison', fontsize=13, fontweight='bold', color=COLORS['primary'])
    plt.tight_layout(); save_figure(fig, f'station_{station_idx:02d}_feature_groups', stage='stage2')


def plot_feature_importance(top_features, station_idx):
    setup_style()
    names = list(top_features.keys()); vals = list(top_features.values())
    fig, ax = plt.subplots(figsize=(10, max(4, len(names)*0.4)))
    colors = [COLORS['accent1'] if any(k in n for k in ['WAV','VMD','EMD'])
              else COLORS['accent2'] if 'IRR' in n else COLORS['accent3'] for n in names]
    ax.barh(names[::-1], vals[::-1], color=colors[::-1], alpha=0.85, height=0.65, edgecolor='white')
    for i, v in enumerate(vals[::-1]):
        if v > 0.01: ax.text(v+0.002, i, f'{v:.1%}', va='center', fontsize=8, color=COLORS['primary'])
    ax.set_xlabel('Feature Importance'); ax.set_title(f'Station {station_idx}: Top Features (All Combined)',
        fontsize=12, fontweight='bold', color=COLORS['primary'])
    plt.tight_layout(); save_figure(fig, f'station_{station_idx:02d}_feature_importance', stage='stage2')


def process_station(filepath, manifest=None):
    fname = os.path.basename(filepath)
    match = re.search(r'station_(\d+)', fname)
    if not match: return None
    station_idx = int(match.group(1))

    print(f"\n{'-'*70}")
    print(f"  Station {station_idx}: {fname}")
    print(f"{'-'*70}")

    df = pd.read_csv(filepath, parse_dates=['DATE_TIME'])
    print(f"  Loaded: {len(df):,} rows x {len(df.columns)} cols")
    if manifest: manifest.log_input(fname, Path(filepath))

    decomp_info = detect_decomposition_cols(df)
    n_decomp = len(decomp_info['all_decomp'])
    decomp_type = ('Wavelet' if any('WAV' in c for c in decomp_info['all_decomp'])
                   else 'EMD' if any('EMD' in c for c in decomp_info['all_decomp'])
                   else 'VMD' if any('VMD' in c for c in decomp_info['all_decomp']) else 'None')
    print(f"  Decomposition: {decomp_type} ({n_decomp} components)")
    print(f"    Trend:  {decomp_info['trend']}")
    print(f"    Daily:  {decomp_info['daily']}")
    print(f"    Hourly: {decomp_info['hourly']}")
    print(f"    Noise:  {decomp_info['noise']}")

    train_idx = np.where(df['SPLIT']=='train')[0]
    test_idx = np.where(df['SPLIT']=='test')[0]
    stage2_cfg = cfg.get('stage2', {})
    target_col = 'TARGET_H1'

    # Base features
    base_cols = ['IRRADIATION', 'TEMPERATURE', 'HOUR', 'DAY', 'MONTH', 'DOY']
    for col in ['REL_HUMIDITY', 'ATMOSPHERE', 'GHI', 'DNI']:
        if col in df.columns: base_cols.append(col)
    base_cols += decomp_info['all_decomp']

    print(f"\n  Base features: {len(base_cols)}")
    t0 = time.time()
    m_base, _ = evaluate_features(df, base_cols, target_col, train_idx, test_idx)
    print(f"    Baseline R^2={m_base['r2']:.6f} ({m_base['n_features']} feat, {time.time()-t0:.1f}s)")

    # Build features
    print(f"\n  Engineering features...")
    t0 = time.time()
    all_new, feature_groups = build_component_features(df, decomp_info, stage2_cfg)

    for col_name, col_values in all_new.items():
        df[col_name] = col_values.values if isinstance(col_values, pd.Series) else col_values

    new_feature_cols = list(all_new.keys())
    for col in new_feature_cols:
        if df[col].isna().any():
            train_mean = df.loc[df['SPLIT']=='train', col].mean()
            df[col] = df[col].fillna(train_mean if not np.isnan(train_mean) else 0.0)

    print(f"    Generated {len(new_feature_cols)} features in {time.time()-t0:.1f}s")
    for gn, gc in feature_groups.items():
        print(f"    {gn}: {len(gc)} features")

    # Evaluate groups
    print(f"\n  Evaluating feature groups...")
    group_results = {'0_base_only': m_base}

    for gn, gc in sorted(feature_groups.items()):
        test_cols = [c for c in base_cols + gc if c in df.columns]
        t0 = time.time()
        m_grp, _ = evaluate_features(df, test_cols, target_col, train_idx, test_idx)
        group_results[gn] = m_grp
        delta = (m_grp['r2'] - m_base['r2']) * 100
        print(f"    + {gn}: R^2={m_grp['r2']:.6f} (delta ={delta:+.4f} pp, {m_grp['n_features']} feat, {time.time()-t0:.1f}s)")

    # All combined
    all_feat = base_cols + new_feature_cols
    seen = set(); all_feat = [c for c in all_feat if c in df.columns and c not in seen and not seen.add(c)]

    t0 = time.time()
    m_all, mdl = evaluate_features(df, all_feat, target_col, train_idx, test_idx)
    group_results['ALL_combined'] = m_all
    delta_all = (m_all['r2'] - m_base['r2']) * 100
    print(f"    + ALL_combined: R^2={m_all['r2']:.6f} (delta ={delta_all:+.4f} pp, {m_all['n_features']} feat, {time.time()-t0:.1f}s)")

    winner_name = max(group_results, key=lambda k: group_results[k]['r2'])
    winner_r2 = group_results[winner_name]['r2']
    delta_pp = (winner_r2 - m_base['r2']) * 100
    print(f"\n  Winner: {winner_name} (R^2={winner_r2:.6f}, delta ={delta_pp:+.4f} pp)")
    print(f"  Keeping ALL {len(all_feat)} features for Stage 3 selection")

    # Plots
    plot_feature_groups(group_results, station_idx)
    if 'top_features' in m_all: plot_feature_importance(m_all['top_features'], station_idx)

    # Save
    meta_cols = ['DATE_TIME', 'DC_POWER', 'SPLIT']
    target_cols = [c for c in df.columns if c.startswith('TARGET_H')]
    out_cols = meta_cols + all_feat + target_cols
    seen = set(); out_cols = [c for c in out_cols if c in df.columns and c not in seen and not seen.add(c)]
    df_out = df[out_cols].copy()

    proc_dir = get_path('data_processed')
    out_name = f'station_{station_idx:02d}_features.csv'
    out_path = proc_dir / out_name
    df_out.to_csv(out_path, index=False)
    print(f"\n  OK Saved: {out_name} ({len(df_out):,} rows x {len(df_out.columns)} cols)")

    if manifest:
        manifest.log_output(out_name, out_path)
        manifest.log_metric(f'station_{station_idx}_base_r2', m_base['r2'])
        manifest.log_metric(f'station_{station_idx}_winner', winner_name)
        manifest.log_metric(f'station_{station_idx}_winner_r2', winner_r2)
        manifest.log_metric(f'station_{station_idx}_n_features', len(all_feat))

    return {
        'station': station_idx, 'decomp_type': decomp_type,
        'base_r2': m_base['r2'], 'winner': winner_name,
        'winner_r2': winner_r2, 'delta_pp': delta_pp,
        'n_features': len(all_feat), 'n_total_cols': len(df_out.columns),
    }


def run(station_idx=None, manifest=None):
    set_all_seeds()
    proc_dir = get_path('data_processed')
    files = sorted(glob.glob(str(proc_dir / 'station_*_decomposed.csv')))
    if not files: raise FileNotFoundError(f"No station_*_decomposed.csv in {proc_dir}. Run Stage 1 first.")
    if station_idx is not None:
        files = [f for f in files if f'station_{station_idx:02d}_' in f]
        if not files: raise FileNotFoundError(f"No decomposed file for station {station_idx}")

    print(f"\n  Stage 2: Component-Specific Feature Engineering")
    print(f"  Found {len(files)} station file(s)")

    all_summaries = []
    for filepath in files:
        summary = process_station(filepath, manifest)
        if summary: all_summaries.append(summary)

    if all_summaries:
        print(f"\n{'-'*70}")
        print(f"  STAGE 2 SUMMARY")
        print(f"{'-'*70}")
        print(f"  {'Station':>8} {'Decomp':<10} {'Base R^2':>10} {'Final R^2':>10} {'Gain':>10} {'Features':>10}")
        print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for s in all_summaries:
            print(f"  {s['station']:>8} {s['decomp_type']:<10} {s['base_r2']:>10.6f} "
                  f"{s['winner_r2']:>10.6f} {s['delta_pp']:>+10.4f} {s['n_features']:>10}")
        save_metrics({'stations': all_summaries}, 'stage2', get_path('outputs_artifacts'))
        valid = [s for s in all_summaries if s['base_r2'] > 0]
        if valid:
            print(f"\n  Mean gain: {np.mean([s['delta_pp'] for s in valid]):+.4f} pp ({len(valid)} valid stations)")

    print(f"\n  - Stage 2 complete")
    return all_summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Feature Engineering")
    parser.add_argument("--station", type=int, default=None)
    args = parser.parse_args()
    from utils.config import manifest as create_manifest
    m = create_manifest("stage2_feature_engineering")
    try:
        run(station_idx=args.station, manifest=m)
    except Exception as e:
        m.log_param("error", str(e)); raise
    finally:
        print(f"  OK Manifest: {m.save()}")
