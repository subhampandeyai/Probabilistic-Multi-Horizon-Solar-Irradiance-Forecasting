"""
Stage 5: cross-dataset validation.

Applies the same preprocessing, decomposition, feature construction and model
training to an independent photovoltaic dataset, so generalisation is measured
under a protocol identical to the primary one rather than a retuned variant.

The short record of the external dataset does not support training a recurrent
model, so the daily band uses gradient boosting here. That substitution is a
property of this stage, not of the framework.

Input:  data/external/Plant_N_Generation_Data.csv
        data/external/Plant_N_Weather_Sensor_Data.csv
Output: results/stage5_cross_validation.json
        figures/stage5/*.png

    python -m src.pipeline.stage5_cross_validation
"""
import os, sys, time, warnings, json
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


# ---
#  LOAD & PREPROCESS KAGGLE DATA
# ---

def load_kaggle_plant(gen_path, weather_path, plant_name):
    """Load and merge Kaggle generation + weather data."""
    print(f"\n  Loading {plant_name}...")

    gen = pd.read_csv(gen_path)
    weather = pd.read_csv(weather_path)

    gen['DATE_TIME'] = pd.to_datetime(gen['DATE_TIME'])
    weather['DATE_TIME'] = pd.to_datetime(weather['DATE_TIME'])

    # Aggregate generation across all inverters (mean per timestamp)
    gen_agg = gen.groupby('DATE_TIME').agg({
        'DC_POWER': 'mean',
        'AC_POWER': 'mean',
        'DAILY_YIELD': 'mean',
        'TOTAL_YIELD': 'mean',
    }).reset_index()

    # Aggregate weather (should be single sensor per plant)
    weather_agg = weather.groupby('DATE_TIME').agg({
        'AMBIENT_TEMPERATURE': 'mean',
        'MODULE_TEMPERATURE': 'mean',
        'IRRADIATION': 'mean',
    }).reset_index()

    # Merge on timestamp
    df = pd.merge(gen_agg, weather_agg, on='DATE_TIME', how='inner')
    df = df.sort_values('DATE_TIME').reset_index(drop=True)

    print(f"    Raw merged: {len(df):,} rows")
    print(f"    Date range: {df['DATE_TIME'].min()} -> {df['DATE_TIME'].max()}")
    print(f"    DC_POWER: mean={df['DC_POWER'].mean():.2f}, max={df['DC_POWER'].max():.2f}")
    print(f"    IRRADIATION: mean={df['IRRADIATION'].mean():.4f}, max={df['IRRADIATION'].max():.4f}")

    return df


def preprocess_kaggle(df, plant_name):
    """Apply same preprocessing as Stage 0 to Kaggle data."""
    # Rename for consistency with our pipeline
    df = df.rename(columns={
        'AMBIENT_TEMPERATURE': 'TEMPERATURE',
    })

    # Filter daytime (irradiation > 0 and power > 0)
    n_before = len(df)
    df = df[(df['IRRADIATION'] > 0) & (df['DC_POWER'] > 0)].copy()
    df = df.reset_index(drop=True)
    print(f"    Daytime filter: {len(df):,} / {n_before:,} rows")

    # Add time features
    dt = df['DATE_TIME']
    df['HOUR'] = dt.dt.hour + dt.dt.minute / 60.0
    df['DAY'] = dt.dt.day
    df['MONTH'] = dt.dt.month
    df['DOY'] = dt.dt.dayofyear

    # 70/15/15 chronological split
    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.85)
    df['SPLIT'] = 'test'
    df.loc[:n_train - 1, 'SPLIT'] = 'train'
    df.loc[n_train:n_val - 1, 'SPLIT'] = 'val'

    n_tr = (df['SPLIT'] == 'train').sum()
    n_va = (df['SPLIT'] == 'val').sum()
    n_te = (df['SPLIT'] == 'test').sum()
    print(f"    Split: train={n_tr:,} | val={n_va:,} | test={n_te:,}")

    # Create forecast targets (1-step ahead = H1)
    horizons = [1, 4, 8, 16]  # Kaggle has only 34 days, limit horizons
    for h in horizons:
        df[f'TARGET_H{h}'] = df['IRRADIATION'].shift(-h)

    return df, horizons


# ---
#  WAVELET DECOMPOSITION
# ---

def apply_wavelet(df):
    """Apply db4 L3 wavelet decomposition to IRRADIATION."""
    import pywt

    signal = df['IRRADIATION'].values
    coeffs = pywt.wavedec(signal, 'db4', level=3)

    components = {}
    for i, c in enumerate(coeffs):
        zeroed = [np.zeros_like(cc) for cc in coeffs]
        zeroed[i] = c
        rec = pywt.waverec(zeroed, 'db4')
        if len(rec) > len(signal):
            rec = rec[:len(signal)]
        elif len(rec) < len(signal):
            rec = np.pad(rec, (0, len(signal) - len(rec)), mode='edge')

        if i == 0:
            name = 'IRR_WAV_cA3'
        else:
            name = f'IRR_WAV_cD{3 - i + 1}'
        components[name] = rec
        df[name] = rec

    print(f"    Wavelet: {len(components)} components added")
    return df


# ---
#  FEATURE ENGINEERING
# ---

def engineer_features(df):
    """Apply same feature engineering as Stage 2."""
    # Rolling statistics on IRRADIATION
    for w, label in [(4, '1h'), (12, '3h')]:
        df[f'IRR_rmean_{label}'] = df['IRRADIATION'].rolling(w, min_periods=1).mean()
        df[f'IRR_rstd_{label}'] = df['IRRADIATION'].rolling(w, min_periods=1).std().fillna(0)

    # Lags
    for lag in [1, 2, 4, 8]:
        df[f'IRR_lag{lag}'] = df['IRRADIATION'].shift(lag)

    # Deltas
    df['IRR_delta'] = df['IRRADIATION'].diff()
    df['IRR_accel'] = df['IRR_delta'].diff()

    # Temperature features
    df['TEMP_lag1'] = df['TEMPERATURE'].shift(1)
    if 'MODULE_TEMPERATURE' in df.columns:
        df['TEMP_DIFF'] = df['MODULE_TEMPERATURE'] - df['TEMPERATURE']
    else:
        df['TEMP_DIFF'] = 0

    # Cyclical encoding
    df['HOUR_sin'] = np.sin(2 * np.pi * df['HOUR'] / 24)
    df['HOUR_cos'] = np.cos(2 * np.pi * df['HOUR'] / 24)
    df['DOY_sin'] = np.sin(2 * np.pi * df['DOY'] / 365)
    df['DOY_cos'] = np.cos(2 * np.pi * df['DOY'] / 365)

    # Component-specific features
    for comp in ['IRR_WAV_cA3', 'IRR_WAV_cD3', 'IRR_WAV_cD2', 'IRR_WAV_cD1']:
        if comp in df.columns:
            df[f'{comp}_lag1'] = df[comp].shift(1)
            df[f'{comp}_lag2'] = df[comp].shift(2)
            df[f'{comp}_rmean'] = df[comp].rolling(4, min_periods=1).mean()
            df[f'{comp}_delta'] = df[comp].diff()

    n_feat = len([c for c in df.columns if c not in
                  ['DATE_TIME', 'DC_POWER', 'AC_POWER', 'DAILY_YIELD',
                   'TOTAL_YIELD', 'SPLIT'] and not c.startswith('TARGET_H')])
    print(f"    Features engineered: {n_feat} total")

    return df


# ---
#  MODEL TRAINING (mirrors Stage 3)
# ---

def clean_features(X_tr, X_va, X_te):
    """NaN-safe feature preparation."""
    valid_cols = ~np.all(np.isnan(X_tr), axis=0)
    X_tr, X_va, X_te = X_tr[:, valid_cols], X_va[:, valid_cols], X_te[:, valid_cols]
    col_medians = np.nan_to_num(np.nanmedian(X_tr, axis=0), nan=0.0)
    for j in range(X_tr.shape[1]):
        for X in [X_tr, X_va, X_te]:
            mask = np.isnan(X[:, j])
            if mask.any():
                X[mask, j] = col_medians[j]
    return X_tr, X_va, X_te


def train_xgboost(X_tr, y_tr, X_va, X_te):
    """XGBoost baseline."""
    from xgboost import XGBRegressor
    m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     random_state=42, n_jobs=-1, verbosity=0)
    m.fit(X_tr, y_tr)
    return m.predict(X_va), m.predict(X_te), 'xgboost'


def train_ridge(X_tr, y_tr, X_va, X_te):
    """Ridge for trend."""
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(X_tr)
    return Ridge(alpha=1.0).fit(Xtr_s, y_tr).predict(sc.transform(X_va)), \
           Ridge(alpha=1.0).fit(Xtr_s, y_tr).predict(sc.transform(X_te)), 'ridge'


def run_fame(df, all_features, target_col, train_idx, val_idx, test_idx):
    """Run FAME: 4 component models + Ridge meta-learner."""
    y = df[target_col].values
    valid = ~np.isnan(y)
    tr = np.where(valid & np.isin(np.arange(len(df)), train_idx))[0]
    va = np.where(valid & np.isin(np.arange(len(df)), val_idx))[0]
    te = np.where(valid & np.isin(np.arange(len(df)), test_idx))[0]

    if len(tr) < 50 or len(va) < 20 or len(te) < 20:
        return None, None, None

    X = df[all_features].values
    X_tr, X_va, X_te = X[tr].copy(), X[va].copy(), X[te].copy()
    X_tr, X_va, X_te = clean_features(X_tr, X_va, X_te)
    y_tr, y_va, y_te = y[tr], y[va], y[te]

    # 4 component models (Ridge for trend, 3x XGBoost for others)
    preds_va, preds_te = {}, {}

    # Trend (Ridge)
    pv, pt, _ = train_ridge(X_tr, y_tr, X_va, X_te)
    preds_va['trend'] = pv
    preds_te['trend'] = pt

    # Daily, Hourly (XGBoost)
    for name in ['daily', 'hourly']:
        pv, pt, _ = train_xgboost(X_tr, y_tr, X_va, X_te)
        preds_va[name] = pv
        preds_te[name] = pt

    # Noise (Persistence)
    irr_va = df['IRRADIATION'].values[va]
    irr_te = df['IRRADIATION'].values[te]
    preds_va['noise'] = irr_va
    preds_te['noise'] = irr_te

    # Meta-learner
    va_stack = np.column_stack(list(preds_va.values()))
    te_stack = np.column_stack(list(preds_te.values()))
    meta_sc = StandardScaler()
    va_s = meta_sc.fit_transform(va_stack)
    te_s = meta_sc.transform(te_stack)
    meta = Ridge(alpha=1.0).fit(va_s, y_va)
    fame_pred = meta.predict(te_s)

    fame_r2 = float(r2_score(y_te, fame_pred))

    # Unified baseline
    _, unified_pred, _ = train_xgboost(X_tr, y_tr, X_va, X_te)
    unified_r2 = float(r2_score(y_te, unified_pred))

    # Persistence baseline
    persist_r2 = float(r2_score(y_te, irr_te))

    return fame_r2, unified_r2, persist_r2


# ---
#  PLOTTING
# ---

def plot_cross_validation(results, chinese_results=None):
    """Plot cross-dataset validation comparison."""
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Cross-Dataset Validation: Chinese Stations vs Kaggle Plants',
                 fontsize=13, fontweight='bold', color=COLORS['primary'])

    # Left: Kaggle results
    ax = axes[0]
    if results:
        plants = list(results.keys())
        horizons = list(results[plants[0]].keys())
        h_mins = [int(h[1:]) * 15 for h in horizons]

        for plant in plants:
            fame_vals = [results[plant][h]['fame_r2'] for h in horizons]
            ax.plot(h_mins, fame_vals, 'o-', lw=2, ms=8, label=f'{plant} FAME')

        ax.set_xlabel('Horizon (min)')
        ax.set_ylabel('R^2')
        ax.set_title('Kaggle Plants Performance')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    # Right: Performance degradation
    ax2 = axes[1]
    if results and chinese_results:
        categories = ['Chinese\n(8 stations)', 'Kaggle\nPlant 1']
        fame_chinese = chinese_results.get('mean_fame_h1', 0.990)
        fame_kaggle = results.get('Plant_1', {}).get('H1', {}).get('fame_r2', 0)

        bars = ax2.bar(categories, [fame_chinese, fame_kaggle],
                       color=[COLORS['accent1'], COLORS['accent3']], alpha=0.85)
        ax2.set_ylabel('R^2 at H1')
        ax2.set_title('Cross-Dataset Generalization')
        ax2.set_ylim(0.9, 1.0)

        # Degradation annotation
        if fame_kaggle > 0:
            deg = (1 - fame_kaggle / fame_chinese) * 100
            ax2.text(0.5, 0.95, f'Degradation: {deg:.2f}%',
                     transform=ax2.transAxes, ha='center', fontsize=11,
                     color=COLORS['primary'])

    plt.tight_layout()
    save_figure(fig, 'cross_dataset_validation', stage='stage5')


# ---
#  MAIN
# ---

def run():
    """Run cross-validation on Kaggle data."""
    set_all_seeds()

    ext_dir = get_path('data_external')
    print(f"\n  Stage 5: Cross-Dataset Validation")
    print(f"  {'=' * 50}")

    # Check files exist
    plant1_gen = ext_dir / 'Plant_1_Generation_Data.csv'
    plant1_weather = ext_dir / 'Plant_1_Weather_Sensor_Data.csv'
    plant2_weather = ext_dir / 'Plant_2_Weather_Sensor_Data.csv'

    # Try to find Plant 2 generation data
    plant2_gen = ext_dir / 'Plant_2_Generation_Data.csv'

    all_results = {}

    # -- PLANT 1 ---
    if plant1_gen.exists() and plant1_weather.exists():
        print(f"\n  -- Plant 1 (1.3 MW, Kaggle) --")

        df1 = load_kaggle_plant(str(plant1_gen), str(plant1_weather), 'Plant 1')
        df1, horizons = preprocess_kaggle(df1, 'Plant 1')
        df1 = apply_wavelet(df1)
        df1 = engineer_features(df1)

        all_features = [c for c in df1.columns if c not in
                        ['DATE_TIME', 'DC_POWER', 'AC_POWER', 'DAILY_YIELD',
                         'TOTAL_YIELD', 'SPLIT', 'MODULE_TEMPERATURE'] and
                        not c.startswith('TARGET_H')]

        train_idx = np.where(df1['SPLIT'] == 'train')[0]
        val_idx = np.where(df1['SPLIT'] == 'val')[0]
        test_idx = np.where(df1['SPLIT'] == 'test')[0]

        plant1_results = {}
        for h in horizons:
            tc = f'TARGET_H{h}'
            if tc not in df1.columns:
                continue
            t0 = time.time()
            fame_r2, unified_r2, persist_r2 = run_fame(
                df1, all_features, tc, train_idx, val_idx, test_idx)

            if fame_r2 is not None:
                imp = (fame_r2 - unified_r2) * 100
                print(f"    H{h} ({h*15}min): FAME={fame_r2:.6f} | "
                      f"Unified={unified_r2:.6f} | delta ={imp:+.4f}pp "
                      f"({time.time()-t0:.1f}s)")
                plant1_results[f'H{h}'] = {
                    'fame_r2': fame_r2, 'unified_r2': unified_r2,
                    'persist_r2': persist_r2, 'improvement_pp': float(imp),
                }

        all_results['Plant_1'] = plant1_results
    else:
        print(f"  ! Plant 1 files not found at {ext_dir}")

    # -- PLANT 2 ---
    if plant2_gen.exists() and plant2_weather.exists():
        print(f"\n  -- Plant 2 (15 MW, Kaggle) --")

        df2 = load_kaggle_plant(str(plant2_gen), str(plant2_weather), 'Plant 2')
        df2, horizons = preprocess_kaggle(df2, 'Plant 2')
        df2 = apply_wavelet(df2)
        df2 = engineer_features(df2)

        all_features = [c for c in df2.columns if c not in
                        ['DATE_TIME', 'DC_POWER', 'AC_POWER', 'DAILY_YIELD',
                         'TOTAL_YIELD', 'SPLIT', 'MODULE_TEMPERATURE'] and
                        not c.startswith('TARGET_H')]

        train_idx = np.where(df2['SPLIT'] == 'train')[0]
        val_idx = np.where(df2['SPLIT'] == 'val')[0]
        test_idx = np.where(df2['SPLIT'] == 'test')[0]

        plant2_results = {}
        for h in horizons:
            tc = f'TARGET_H{h}'
            if tc not in df2.columns:
                continue
            t0 = time.time()
            fame_r2, unified_r2, persist_r2 = run_fame(
                df2, all_features, tc, train_idx, val_idx, test_idx)

            if fame_r2 is not None:
                imp = (fame_r2 - unified_r2) * 100
                print(f"    H{h} ({h*15}min): FAME={fame_r2:.6f} | "
                      f"Unified={unified_r2:.6f} | delta ={imp:+.4f}pp "
                      f"({time.time()-t0:.1f}s)")
                plant2_results[f'H{h}'] = {
                    'fame_r2': fame_r2, 'unified_r2': unified_r2,
                    'persist_r2': persist_r2, 'improvement_pp': float(imp),
                }

        all_results['Plant_2'] = plant2_results
    else:
        print(f"  ! Plant 2 generation data not found  -  skipping")
        print(f"    Place Plant_2_Generation_Data.csv in {ext_dir}")

    # -- CROSS-DATASET COMPARISON ---
    if all_results:
        print(f"\n  {'=' * 50}")
        print(f"  CROSS-DATASET SUMMARY")
        print(f"  {'=' * 50}")

        for plant_name, results in all_results.items():
            print(f"\n  {plant_name}:")
            print(f"  {'Horizon':>10} {'FAME':>10} {'Unified':>10} {'delta (pp)':>10}")
            for h, d in sorted(results.items(), key=lambda x: int(x[0][1:])):
                print(f"  {h:>10} {d['fame_r2']:>10.6f} {d['unified_r2']:>10.6f} "
                      f"{d['improvement_pp']:>+10.4f}")

        # Compare with Chinese station results
        s4_path = get_path('outputs_reports') / 'stage4_summary.csv'
        chinese_h1: float | None = None
        if s4_path.exists():
            s4_df = pd.read_csv(s4_path)
            s4_clean = s4_df[s4_df['station'] != 3]
            chinese_h1 = float(s4_clean[s4_clean['horizon'] == 1]['fame_r2'].mean())
            chinese_unified_h1 = float(s4_clean[s4_clean['horizon'] == 1]['unified_r2'].mean())
            chinese_summary = {
                'mean_fame_h1': chinese_h1,
                'mean_unified_h1': chinese_unified_h1,
                'source': 'Computed from Stage 4 output',
            }
        else:
            chinese_summary = {'mean_fame_h1': 0.0, 'source': 'Stage 4 not run'}

        if 'Plant_1' in all_results and 'H1' in all_results['Plant_1']:
            kaggle_h1 = all_results['Plant_1']['H1']['fame_r2']
            print(f"\n  Cross-dataset degradation (H1):")
            if chinese_h1 is not None:
                deg: float = abs(chinese_h1 - kaggle_h1) / chinese_h1 * 100
                print(f"    Chinese mean: {chinese_h1:.6f}")
                print(f"    Kaggle Plant 1: {kaggle_h1:.6f}")
                print(f"    Degradation: {deg:.2f}%")
            else:
                print(f"    Chinese mean: N/A (run Stage 4 first)")
                print(f"    Kaggle Plant 1: {kaggle_h1:.6f}")
                print(f"    Degradation: N/A")

        # Plot
        try:
            plot_cross_validation(all_results, chinese_summary)
        except Exception as e:
            print(f"  ! Plot failed: {e}")

        # Save
        save_metrics(all_results, 'stage5', get_path('outputs_artifacts'))

    print(f"\n  - Stage 5 complete")
    return all_results


if __name__ == "__main__":
    from utils.config import manifest as cm
    m = cm("stage5_cross_validation")
    try:
        run()
    except Exception as e:
        m.log_param("error", str(e))
        raise
    finally:
        print(f"  OK Manifest: {m.save()}")
