"""
Stage 3: frequency-aligned base learners and Ridge meta-learner.

Each decomposition band is modelled by the learner matched to its spectral
character, and the four component forecasts are fused by an L2-regularised
Ridge meta-learner fitted on the validation split:

    trend  -> Ridge       (smooth, slowly varying envelope)
    daily  -> LSTM        (cyclical temporal dependence)
    hourly -> XGBoost     (feature-driven medium-scale variation)
    noise  -> Persistence (lowest predictability scale)

All learners receive the same NaN-free feature matrix, so any difference in
their predictions reflects architecture rather than unequal inputs. Ridge and
LSTM cannot accept NaN, so features are imputed before training.
"""
import os, sys, glob, time, argparse, warnings, json, re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import cfg, get_path, set_all_seeds
from utils.metrics import compute_all, save_metrics, bootstrap_ci
from utils.plotting import setup_style, save_figure, COLORS
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# -- TensorFlow check at module level ---
TF_AVAILABLE = False
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    TF_AVAILABLE = True
    TF_VERSION = tf.__version__
except ImportError:
    TF_VERSION = None


# ---
#  NaN-SAFE FEATURE PREPARATION
# ---

def clean_features(X_tr, X_va, X_te):
    """
    Remove columns that are entirely NaN, then impute remaining NaN
    with column median from training set. This is critical for Ridge
    and LSTM which cannot handle NaN.
    """
    # Find columns with any valid data in train
    valid_cols = ~np.all(np.isnan(X_tr), axis=0)
    X_tr = X_tr[:, valid_cols]
    X_va = X_va[:, valid_cols]
    X_te = X_te[:, valid_cols]

    # Compute column medians from training data
    col_medians = np.nanmedian(X_tr, axis=0)
    # Replace any remaining NaN medians with 0
    col_medians = np.nan_to_num(col_medians, nan=0.0)

    # Impute NaN with column medians
    for j in range(X_tr.shape[1]):
        mask_tr = np.isnan(X_tr[:, j])
        mask_va = np.isnan(X_va[:, j])
        mask_te = np.isnan(X_te[:, j])
        if mask_tr.any():
            X_tr[mask_tr, j] = col_medians[j]
        if mask_va.any():
            X_va[mask_va, j] = col_medians[j]
        if mask_te.any():
            X_te[mask_te, j] = col_medians[j]

    return X_tr, X_va, X_te


# ---
#  COMPONENT CLASSIFICATION
# ---

def classify_components(df):
    """Classify raw decomposition columns into frequency bands."""
    comp_cols = [c for c in df.columns
                 if (c.startswith('IRR_WAV_c') or c.startswith('IRR_EMD_IMF')
                     or c.startswith('IRR_VMD_M'))
                 and not any(s in c for s in
                     ['_lag', '_rmean', '_rstd', '_rmin', '_rmax', '_delta', '_accel'])]
    if not comp_cols:
        return {}
    bands = {'trend': [], 'daily': [], 'hourly': [], 'noise': []}
    for col in comp_cols:
        if 'cA' in col or col == 'IRR_VMD_M1':
            bands['trend'].append(col)
        elif 'cD3' in col or 'cD4' in col or col == 'IRR_VMD_M2':
            bands['daily'].append(col)
        elif 'cD2' in col or col in ('IRR_VMD_M3', 'IRR_VMD_M4'):
            bands['hourly'].append(col)
        else:
            bands['noise'].append(col)
    return {k: v for k, v in bands.items() if v}


def get_all_features(df):
    """All feature columns  -  every model sees everything."""
    exclude = {'DATE_TIME', 'DC_POWER', 'SPLIT'}
    return [c for c in df.columns
            if c not in exclude and not c.startswith('TARGET_H')]


# ---
#  MODEL BUILDERS
#  Each returns (pred_val, pred_test, model_type_str)
#  All receive CLEAN (NaN-free) feature arrays
# ---

def train_ridge(X_tr, y_tr, X_va, X_te):
    """Ridge regression for smooth trend."""
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(X_tr)
    Xva_s = sc.transform(X_va)
    Xte_s = sc.transform(X_te)
    m = Ridge(alpha=cfg['stage3']['ridge']['alpha'])
    m.fit(Xtr_s, y_tr)
    return m.predict(Xva_s), m.predict(Xte_s), 'ridge'


def train_xgboost(X_tr, y_tr, X_va, X_te):
    """XGBoost for feature-driven hourly variation."""
    from xgboost import XGBRegressor
    p = cfg['stage3']['xgboost']
    m = XGBRegressor(
        n_estimators=p.get('n_estimators', 800),
        max_depth=p.get('max_depth', 6),
        learning_rate=p.get('learning_rate', 0.03),
        subsample=p.get('subsample', 0.8),
        colsample_bytree=p.get('colsample_bytree', 0.8),
        reg_alpha=p.get('reg_alpha', 0.1),
        reg_lambda=p.get('reg_lambda', 1.0),
        random_state=cfg['seeds']['global'], n_jobs=-1, verbosity=0)
    m.fit(X_tr, y_tr)
    return m.predict(X_va), m.predict(X_te), 'xgboost'


def train_lstm(X_tr, y_tr, X_va, y_va, X_te):
    """LSTM for cyclical daily patterns.

    CORRECTNESS REQUIREMENT: the daily band must run an LSTM. If it falls back
    to XGBoost it fits the same model on the same shared features as the hourly
    band, the two bands produce identical predictions, and the meta-learner
    splits a single weight arbitrarily between them -- which makes the per-band
    coefficients meaningless. stage3.lstm.require_tensorflow keeps the run from
    proceeding in that state; any fallback that is allowed is tagged in the
    returned model-type string so downstream analysis can detect it.
    """
    strict = cfg['stage3'].get('lstm', {}).get('require_tensorflow', False)

    if not TF_AVAILABLE:
        if strict:
            raise RuntimeError(
                "TensorFlow is required for the daily (LSTM) band but is not "
                "installed. Install TensorFlow, or set "
                "stage3.lstm.require_tensorflow: false to accept an XGBoost "
                "fallback, which makes the daily and hourly bands identical.")
        print("        (TensorFlow unavailable, using XGBoost for this band)")
        return (*train_xgboost(X_tr, y_tr, X_va, X_te)[:2], 'xgboost_substitute_no_tf')

    lc = cfg['stage3']['lstm']
    seq_len = min(lc.get('sequence_length', 48), len(X_tr) // 4)
    if seq_len < 8 or len(X_tr) < 200:
        if strict:
            raise RuntimeError(
                f"Too few training samples ({len(X_tr)}) for the LSTM band.")
        print("        (Sequence too short, using XGBoost for this band)")
        return (*train_xgboost(X_tr, y_tr, X_va, X_te)[:2], 'xgboost_substitute_short')

    tf.random.set_seed(cfg['seeds']['global'])

    # Scale (features are already NaN-free from clean_features)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(X_tr)
    Xva_s = sc.transform(X_va)
    Xte_s = sc.transform(X_te)
    nf = Xtr_s.shape[1]

    def make_seq(X, y, sl):
        Xs, ys = [], []
        for i in range(sl, len(X)):
            Xs.append(X[i - sl:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    # Training sequences
    Xseq_tr, yseq_tr = make_seq(Xtr_s, y_tr, seq_len)

    # Build model
    units = lc.get('units', [128, 64])
    mdl = tf.keras.Sequential()
    mdl.add(tf.keras.layers.LSTM(units[0], input_shape=(seq_len, nf),
                                  return_sequences=(len(units) > 1)))
    mdl.add(tf.keras.layers.Dropout(lc.get('dropout', 0.2)))
    if len(units) > 1:
        mdl.add(tf.keras.layers.LSTM(units[1]))
        mdl.add(tf.keras.layers.Dropout(lc.get('dropout', 0.2)))
    mdl.add(tf.keras.layers.Dense(32, activation='relu'))
    mdl.add(tf.keras.layers.Dense(1))
    mdl.compile(optimizer='adam', loss='mse')

    mdl.fit(Xseq_tr, yseq_tr,
            epochs=lc.get('epochs', 100),
            batch_size=lc.get('batch_size', 64),
            validation_split=0.15, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                patience=lc.get('patience', 10),
                restore_best_weights=True)])

    # Validation prediction (with context from end of train)
    X_va_ctx = np.vstack([Xtr_s[-seq_len:], Xva_s])
    y_va_ctx = np.concatenate([y_tr[-seq_len:], y_va])
    Xseq_va, _ = make_seq(X_va_ctx, y_va_ctx, seq_len)
    pred_va_seq = mdl.predict(Xseq_va, verbose=0).flatten()
    pred_va = np.full(len(X_va), float(np.mean(pred_va_seq)))
    n = min(len(pred_va_seq), len(X_va))
    pred_va[:n] = pred_va_seq[:n]

    # Test prediction (with context from end of train)
    X_te_ctx = np.vstack([Xtr_s[-seq_len:], Xte_s])
    y_te_ctx = np.zeros(len(X_te_ctx))
    Xseq_te, _ = make_seq(X_te_ctx, y_te_ctx, seq_len)
    pred_te_seq = mdl.predict(Xseq_te, verbose=0).flatten()
    pred_te = np.full(len(X_te), float(np.mean(pred_te_seq)))
    n = min(len(pred_te_seq), len(X_te))
    pred_te[:n] = pred_te_seq[:n]

    return pred_va, pred_te, 'lstm'


# ---
#  CORE: PROCESS ONE HORIZON
# ---

def run_horizon(df, bands, all_features, target_col, horizon,
                train_idx, val_idx, test_idx, model_map):
    """
    For one forecast horizon:
      1. Clean features (impute NaN)
      2. Train each component's matched model on ALL features
      3. Ridge meta-learner combines on validation set
      4. Final prediction on test set
    """
    y = df[target_col].values
    valid = ~np.isnan(y)

    tr = np.where(valid & np.isin(np.arange(len(df)), train_idx))[0]
    va = np.where(valid & np.isin(np.arange(len(df)), val_idx))[0]
    te = np.where(valid & np.isin(np.arange(len(df)), test_idx))[0]

    if len(tr) < 100 or len(va) < 50 or len(te) < 50:
        return None

    # Extract raw feature arrays
    X_raw = df[all_features].values
    X_tr_raw = X_raw[tr].copy()
    X_va_raw = X_raw[va].copy()
    X_te_raw = X_raw[te].copy()

    # Clean NaN ONCE for all models
    X_tr_clean, X_va_clean, X_te_clean = clean_features(
        X_tr_raw, X_va_raw, X_te_raw)

    y_tr, y_va, y_te = y[tr], y[va], y[te]
    irr_va = df['IRRADIATION'].values[va]
    irr_te = df['IRRADIATION'].values[te]

    n_feat = X_tr_clean.shape[1]
    preds_va = {}
    preds_te = {}
    details = {}

    for band_name, band_cols in bands.items():
        mt = model_map.get(band_name, 'xgboost')
        t0 = time.time()

        try:
            if mt == 'persistence':
                pv, pt, actual_mt = irr_va.copy(), irr_te.copy(), 'persistence'

            elif mt == 'lstm':
                pv, pt, actual_mt = train_lstm(
                    X_tr_clean, y_tr, X_va_clean, y_va, X_te_clean)

            elif mt == 'ridge':
                pv, pt, actual_mt = train_ridge(
                    X_tr_clean, y_tr, X_va_clean, X_te_clean)

            else:  # xgboost (handles NaN natively, but clean data is fine too)
                pv, pt, actual_mt = train_xgboost(
                    X_tr_clean, y_tr, X_va_clean, X_te_clean)

            elapsed = time.time() - t0

            # Ensure correct lengths
            if len(pv) != len(y_va):
                fix = np.full(len(y_va), float(np.nanmean(pv)))
                fix[:min(len(pv), len(y_va))] = pv[:min(len(pv), len(y_va))]
                pv = fix
            if len(pt) != len(y_te):
                fix = np.full(len(y_te), float(np.nanmean(pt)))
                fix[:min(len(pt), len(y_te))] = pt[:min(len(pt), len(y_te))]
                pt = fix

            # Final NaN safety
            pv = np.nan_to_num(pv, nan=float(np.nanmean(y_tr)))
            pt = np.nan_to_num(pt, nan=float(np.nanmean(y_tr)))

            cr2 = float(r2_score(y_te, pt))
            preds_va[band_name] = pv
            preds_te[band_name] = pt
            details[band_name] = {
                'model_type': actual_mt, 'r2': cr2,
                'n_features': n_feat, 'time_s': round(elapsed, 1)
            }
            print(f"      {band_name} -> {actual_mt}: R^2={cr2:.6f} "
                  f"({n_feat} feat, {elapsed:.1f}s)")

        except Exception as e:
            print(f"      {band_name} -> {mt}: FAIL {e}")
            preds_va[band_name] = irr_va.copy()
            preds_te[band_name] = irr_te.copy()
            details[band_name] = {
                'model_type': 'fallback', 'r2': 0, 'n_features': 0, 'time_s': 0
            }

    if not preds_te:
        return None

    # -- Meta-learner: Ridge stacking ---
    va_stack = np.column_stack(list(preds_va.values()))
    te_stack = np.column_stack(list(preds_te.values()))

    meta_sc = StandardScaler()
    va_s = meta_sc.fit_transform(va_stack)
    te_s = meta_sc.transform(te_stack)

    meta = Ridge(alpha=1.0)
    meta.fit(va_s, y_va)

    fame_pred = meta.predict(te_s)
    fame_r2 = float(r2_score(y_te, fame_pred))
    fame_met = compute_all(y_te, fame_pred)
    ci = bootstrap_ci(y_te, fame_pred)

    weights = dict(zip(preds_te.keys(),
                       [round(float(w), 4) for w in meta.coef_]))
    print(f"    -> FAME: R^2={fame_r2:.6f} [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"       Weights: {weights}")

    return {
        'fame_r2': fame_r2, 'fame_metrics': fame_met, 'fame_ci': ci,
        'y_test': y_te, 'fame_pred': fame_pred,
        'component_details': details, 'meta_weights': weights,
    }


# ---
#  BASELINES
# ---

def run_baselines(df, all_features, target_col, train_idx, test_idx):
    """Unified XGBoost + Persistence."""
    y = df[target_col].values
    valid = ~np.isnan(y)
    tr = np.where(valid & np.isin(np.arange(len(df)), train_idx))[0]
    te = np.where(valid & np.isin(np.arange(len(df)), test_idx))[0]
    X = df[all_features].values

    # XGBoost handles NaN natively, but clean for consistency
    X_tr_c, _, X_te_c = clean_features(
        X[tr].copy(), X[tr].copy(), X[te].copy())

    _, pu, _ = train_xgboost(X_tr_c, y[tr], X_tr_c, X_te_c)
    ur2 = float(r2_score(y[te], pu))
    pr2 = float(r2_score(y[te], df['IRRADIATION'].values[te]))
    return ur2, pr2


# ---
#  PLOTTING
# ---

def plot_horizon_comparison(results, sidx):
    """Multi-horizon R^2 comparison."""
    setup_style()
    hs = sorted(results.keys(), key=lambda x: int(x[1:]))
    hm = [int(h[1:]) * 15 for h in hs]
    fr = [results[h].get('fame_r2') or 0 for h in hs]
    ur = [results[h]['unified_r2'] for h in hs]
    pr = [results[h]['persist_r2'] for h in hs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Station {sidx}: Multi-Horizon Forecast Comparison',
                 fontsize=13, fontweight='bold', color=COLORS['primary'])

    ax = axes[0]
    ax.plot(hm, fr, 'o-', color=COLORS['accent1'], lw=2, ms=8,
            label='FAME', zorder=5)
    ax.plot(hm, ur, 's--', color=COLORS['accent3'], lw=1.5, ms=6,
            label='Unified XGBoost')
    ax.plot(hm, pr, '^:', color=COLORS['neutral'], lw=1, ms=5,
            label='Persistence')
    ax.set_xlabel('Horizon (minutes)')
    ax.set_ylabel('R^2')
    ax.set_title('R^2 vs Forecast Horizon')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    imp = [(f - u) * 100 for f, u in zip(fr, ur)]
    cols = [COLORS['accent2'] if i > 0 else COLORS['accent1'] for i in imp]
    ax2.bar(range(len(hs)), imp, color=cols, alpha=0.8)
    ax2.axhline(y=0, color=COLORS['primary'], lw=1)
    ax2.set_xticks(range(len(hs)))
    ax2.set_xticklabels([f'{m}m' for m in hm], fontsize=8)
    ax2.set_ylabel('FAME - Unified (pp)')
    ax2.set_title('Improvement over Unified XGBoost')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_figure(fig, f'station_{sidx:02d}_horizon_comparison', stage='stage3')


# ---
#  PROCESS ONE STATION
# ---

def process_station(filepath, manifest=None):
    """Full FAME pipeline for one station."""
    fname = os.path.basename(filepath)
    match = re.search(r'station_(\d+)', fname)
    if not match:
        return None
    sidx = int(match.group(1))

    print(f"\n{'=' * 70}")
    print(f"  Station {sidx}: {fname}")
    print(f"{'=' * 70}")

    df = pd.read_csv(filepath, parse_dates=['DATE_TIME'])
    print(f"  Loaded: {len(df):,} rows x {len(df.columns)} cols")

    if manifest:
        manifest.log_input(fname, Path(filepath))

    train_idx = np.where(df['SPLIT'] == 'train')[0]
    val_idx = np.where(df['SPLIT'] == 'val')[0]
    test_idx = np.where(df['SPLIT'] == 'test')[0]

    bands = classify_components(df)
    model_map = cfg['stage3']['component_model_map']
    horizons = cfg['forecasting']['horizons']
    all_features = get_all_features(df)

    # Check for NaN in features
    nan_count = df[all_features].isna().sum().sum()
    print(f"  Features: {len(all_features)} ({nan_count} NaN values -> will be imputed)")
    print(f"  Bands: {list(bands.keys())}")
    for b, cols in bands.items():
        print(f"    {b} -> {model_map.get(b, 'xgb')}: {cols}")

    station_results = {}

    for horizon in horizons:
        tc = f'TARGET_H{horizon}'
        if tc not in df.columns:
            continue

        print(f"\n  -- H{horizon} ({horizon * 15}min) --")
        print(f"    FAME:")

        fame = run_horizon(df, bands, all_features, tc, horizon,
                           train_idx, val_idx, test_idx, model_map)

        t0 = time.time()
        ur2, pr2 = run_baselines(df, all_features, tc, train_idx, test_idx)
        print(f"    Baselines: Unified={ur2:.6f} | Persist={pr2:.6f} "
              f"({time.time()-t0:.1f}s)")

        if fame:
            imp = (fame['fame_r2'] - ur2) * 100
            frmse = fame['fame_metrics']['rmse']
            yte = fame['y_test']
            ite = df['IRRADIATION'].values[
                np.where(~np.isnan(df[tc].values) &
                         np.isin(np.arange(len(df)), test_idx))[0]]
            prmse = float(np.sqrt(np.mean((yte - ite[:len(yte)])**2)))
            fss = 1 - frmse / prmse if prmse > 0 else 0

            print(f"    FAME vs Unified: {imp:+.4f} pp | FSS: {fss:.4f}")

            # Record the learner each band actually used. Two bands running the
            # same model on the same features produce identical predictions, so
            # the meta-weight split between them carries no information; the
            # flag lets downstream analysis reject such a run.
            used = {b: d.get('model_type')
                    for b, d in (fame['component_details'] or {}).items()}
            substituted = any(str(m).startswith('xgboost_substitute')
                              for m in used.values())

            station_results[f'H{horizon}'] = {
                'fame_r2': fame['fame_r2'], 'unified_r2': ur2,
                'persist_r2': pr2, 'improvement_pp': float(imp),
                'fss': float(fss),
                'components': fame['component_details'],
                'meta_weights': fame['meta_weights'],
                'learners_used': used,
                'substituted_bands': substituted,
                'tensorflow_available': TF_AVAILABLE,
                'tensorflow_version': TF_VERSION,
            }
            if substituted:
                print("    ! a band used a substitute learner; per-band "
                      "coefficients are not interpretable for this run")
        else:
            station_results[f'H{horizon}'] = {
                'fame_r2': None, 'unified_r2': ur2, 'persist_r2': pr2,
                'improvement_pp': None, 'fss': None,
            }

    # -- Summary table ---
    print(f"\n  {'-' * 60}")
    print(f"  Station {sidx} Summary")
    print(f"  {'Horizon':>10} {'FAME':>10} {'Unified':>10} {'delta (pp)':>10} {'FSS':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for h in sorted(station_results.keys(), key=lambda x: int(x[1:])):
        d = station_results[h]
        fs = f"{d['fame_r2']:.6f}" if d['fame_r2'] is not None else " - "
        ii = f"{d['improvement_pp']:+.4f}" if d['improvement_pp'] is not None else " - "
        ss = f"{d['fss']:.4f}" if d['fss'] is not None else " - "
        print(f"  {h:>10} {fs:>10} {d['unified_r2']:>10.6f} {ii:>10} {ss:>8}")

    try:
        plot_horizon_comparison(station_results, sidx)
    except Exception as e:
        print(f"  ! Plot: {e}")

    # Save results
    md = get_path('outputs_models')
    rpath = md / f'station_{sidx:02d}_results.json'
    with open(rpath, 'w') as f:
        json.dump(station_results, f, indent=2, default=str)
    print(f"\n  OK Saved: {rpath.name}")

    if manifest:
        for hk, d in station_results.items():
            if d['fame_r2'] is not None:
                manifest.log_metric(f's{sidx}_{hk}_fame', d['fame_r2'])
                manifest.log_metric(f's{sidx}_{hk}_delta', d['improvement_pp'])

    return {'station': sidx, 'results': station_results}


# ---
#  MAIN
# ---

def run(station_idx=None, manifest=None):
    """Process all or one station."""
    set_all_seeds()
    proc = get_path('data_processed')
    files = sorted(glob.glob(str(proc / 'station_*_features.csv')))

    if not files:
        raise FileNotFoundError("No features files found. Run Stage 2 first.")
    if station_idx is not None:
        files = [f for f in files if f'station_{station_idx:02d}_' in f]

    print(f"\n  Stage 3: Frequency-Aligned Model Training")
    print(f"  Models: {cfg['stage3']['component_model_map']}")
    print(f"  Horizons: {cfg['forecasting']['horizons']}")
    if TF_AVAILABLE:
        print(f"  TensorFlow: {TF_VERSION} (LSTM band enabled)")
    else:
        print(f"  ! TensorFlow not found; the daily band cannot run an LSTM")
        print(f"    Install it with: pip install -r requirements.txt")

    summaries = []
    for fp in files:
        s = process_station(fp, manifest)
        if s:
            summaries.append(s)

    # -- Cross-station summary ---
    if summaries:
        print(f"\n{'=' * 70}")
        print(f"  CROSS-STATION RESULTS")
        print(f"{'=' * 70}")

        print(f"\n  H1 (15min ahead):")
        print(f"  {'Stn':>5} {'FAME':>10} {'Unified':>10} {'delta (pp)':>10} {'FSS':>8}")
        for s in summaries:
            h1 = s['results'].get('H1', {})
            if h1.get('fame_r2') is not None:
                print(f"  {s['station']:>5} {h1['fame_r2']:>10.6f} "
                      f"{h1['unified_r2']:>10.6f} "
                      f"{h1['improvement_pp']:>+10.4f} "
                      f"{h1.get('fss', 0):>8.4f}")

        print(f"\n  H96 (24hr ahead):")
        print(f"  {'Stn':>5} {'FAME':>10} {'Unified':>10} {'delta (pp)':>10} {'FSS':>8}")
        for s in summaries:
            h96 = s['results'].get('H96', {})
            if h96.get('fame_r2') is not None:
                print(f"  {s['station']:>5} {h96['fame_r2']:>10.6f} "
                      f"{h96['unified_r2']:>10.6f} "
                      f"{h96['improvement_pp']:>+10.4f} "
                      f"{h96.get('fss', 0):>8.4f}")

        # Mean improvements
        all_imp = []
        for s in summaries:
            for h, d in s['results'].items():
                if d.get('improvement_pp') is not None:
                    all_imp.append(d['improvement_pp'])
        if all_imp:
            print(f"\n  Mean improvement: {np.mean(all_imp):+.4f} pp "
                  f"(across {len(all_imp)} station-horizon pairs)")
            print(f"  Positive: {sum(1 for x in all_imp if x > 0)}/{len(all_imp)} "
                  f"({sum(1 for x in all_imp if x > 0)/len(all_imp)*100:.0f}%)")

        save_metrics({s['station']: s['results'] for s in summaries},
                     'stage3', get_path('outputs_artifacts'))

    print(f"\n  - Stage 3 complete")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 3: Frequency-Adaptive Model Training")
    parser.add_argument("--station", type=int, default=None,
                       help="Process specific station (1-8)")
    args = parser.parse_args()

    from utils.config import manifest as cm
    m = cm("stage3_model_training")
    try:
        run(station_idx=args.station, manifest=m)
    except Exception as e:
        m.log_param("error", str(e))
        raise
    finally:
        print(f"  OK Manifest: {m.save()}")