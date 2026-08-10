"""
Stage 0: preprocessing.

Loads the raw station workbooks, replaces sentinel values with missing entries,
converts units, removes night-time samples, derives temporal features, applies
the chronological train/validation/test split, and emits the multi-horizon
forecast targets. Column names are resolved by keyword from config.yaml, so a
differently named source file can be used without editing this module.

All imputation statistics are computed on the training partition only and then
applied to validation and test, which keeps the split free of information from
later periods.

Input:  data/raw/*.xlsx
Output: data/processed/station_XX_prepared.csv
        outputs/artifacts/stage0_preprocessing_latest.json
        figures/stage0/*.png

    python -m src.pipeline.stage0_preprocessing              # all stations
    python -m src.pipeline.stage0_preprocessing --station 5  # one station
"""
import os, sys, glob, time, argparse, warnings
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# -- Project imports ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import cfg, get_path, set_all_seeds, RunManifest
from utils.schema import validate_dataframe, check_temporal_leakage, SchemaError
from utils.metrics import save_metrics
from utils.plotting import setup_style, save_figure, COLORS


# ---
#  PURE FUNCTIONS
# ---

def find_column(df, keywords):
    """Find first column matching any keyword (case-insensitive)."""
    for kw in keywords:
        for c in df.columns:
            if kw.lower() in str(c).lower():
                return c
    return None


def load_station(filepath, station_cfg):
    """
    Load a single station Excel file and normalize column names.
    Handles naming variations across the 8 stations.
    """
    df_raw = pd.read_excel(filepath, header=0)

    # Detect columns flexibly
    time_col = find_column(df_raw, ['time', 'date'])
    irr_col  = find_column(df_raw, station_cfg.get('irr_col_keywords', ['total solar']))
    temp_col = find_column(df_raw, station_cfg.get('temp_col_keywords', ['air temp', 'temperature']))
    pwr_col  = find_column(df_raw, station_cfg.get('power_col_keywords', ['power']))
    rh_col   = find_column(df_raw, ['humidity'])
    atm_col  = find_column(df_raw, ['atmosphere'])
    ghi_col  = find_column(df_raw, ['global hori'])
    dni_col  = find_column(df_raw, ['direct normal'])

    if not all([time_col, irr_col, temp_col, pwr_col]):
        missing = []
        if not time_col: missing.append('TIME')
        if not irr_col: missing.append('IRRADIATION')
        if not temp_col: missing.append('TEMPERATURE')
        if not pwr_col: missing.append('POWER')
        raise ValueError(f"Missing critical columns: {missing}. Found: {list(df_raw.columns)}")

    # Rename to standard names
    rename_map = {
        time_col: 'DATE_TIME',
        irr_col: 'IRRADIATION',
        temp_col: 'TEMPERATURE',
        pwr_col: 'DC_POWER',
    }
    if rh_col:  rename_map[rh_col] = 'REL_HUMIDITY'
    if atm_col: rename_map[atm_col] = 'ATMOSPHERE'
    if ghi_col: rename_map[ghi_col] = 'GHI'
    if dni_col: rename_map[dni_col] = 'DNI'

    df = df_raw.rename(columns=rename_map).copy()

    return df


def clean_sentinels(df, sentinels=None):
    """Replace sentinel values with NaN, remove physically impossible rows."""
    if sentinels is None:
        sentinels = [-99, -9999, 'null', 'NULL', '']

    numeric_cols = ['DC_POWER', 'IRRADIATION', 'TEMPERATURE']
    extra_cols = ['REL_HUMIDITY', 'ATMOSPHERE', 'GHI', 'DNI']
    all_cols = [c for c in numeric_cols + extra_cols if c in df.columns]

    n_before = len(df)

    for col in all_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].replace({s: np.nan for s in sentinels if isinstance(s, str)}),
                errors='coerce'
            )
        df.loc[df[col] <= -90, col] = np.nan     # no more type error

    # Negative irradiation is physically impossible
    if 'IRRADIATION' in df.columns:
        df.loc[df['IRRADIATION'] < 0, 'IRRADIATION'] = np.nan

    # Negative power is physically impossible for PV
    if 'DC_POWER' in df.columns:
        df.loc[df['DC_POWER'] < 0, 'DC_POWER'] = np.nan

    # Temperature sanity: remove if outside [-60, 60]  degC
    if 'TEMPERATURE' in df.columns:
        df.loc[(df['TEMPERATURE'] < -60) | (df['TEMPERATURE'] > 60), 'TEMPERATURE'] = np.nan

    n_cleaned = df[all_cols].isna().sum().sum()

    return df, n_cleaned


def convert_units(df):
    """
    Auto-detect units and convert:
      - Irradiation: W/m^2 -> kW/m^2 (divide by 1000)
      - Power: MW -> kW (multiply by 1000)
    Detection based on 99.9th percentile of valid values.
    """
    info = {}

    # Irradiation
    irr_valid = df['IRRADIATION'].dropna()
    if len(irr_valid) > 0:
        irr_q999 = irr_valid.quantile(0.999)
        if irr_q999 > 5:  # W/m^2 scale (typical max ~1200 W/m^2)
            df['IRRADIATION'] = df['IRRADIATION'] / 1000.0
            info['irr_conversion'] = f"W/m^2 -> kW/m^2 (max was {irr_q999:.0f})"
        else:
            info['irr_conversion'] = f"Already kW/m^2 (max={irr_q999:.4f})"

    # Convert GHI and DNI too if present
    for col in ['GHI', 'DNI']:
        if col in df.columns:
            valid = df[col].dropna()
            if len(valid) > 0 and valid.quantile(0.999) > 5:
                df[col] = df[col] / 1000.0

    # Power
    pwr_valid = df['DC_POWER'].dropna()
    if len(pwr_valid) > 0:
        pwr_q999 = pwr_valid.quantile(0.999)
        if pwr_q999 < 1000:  # MW scale (typical max 30-130 MW)
            df['DC_POWER'] = df['DC_POWER'] * 1000.0
            info['pwr_conversion'] = f"MW -> kW (max was {pwr_q999:.1f} MW)"
        else:
            info['pwr_conversion'] = f"Already kW (max={pwr_q999:.0f})"

    return df, info


def parse_timestamps(df):
    """Parse DATE_TIME column and sort chronologically."""
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], errors='coerce')
    n_failed = df['DATE_TIME'].isna().sum()
    df = df.dropna(subset=['DATE_TIME']).sort_values('DATE_TIME').reset_index(drop=True)
    return df, n_failed


def add_time_features(df):
    """Add temporal features from DATE_TIME."""
    dt = df['DATE_TIME']
    df['HOUR'] = dt.dt.hour + dt.dt.minute / 60.0
    df['DAY'] = dt.dt.day
    df['MONTH'] = dt.dt.month
    df['DOY'] = dt.dt.dayofyear
    return df


def handle_gaps(df):
    """Detect and interpolate small gaps (up to 1 hour = 4 steps)."""
    df = df.set_index('DATE_TIME').sort_index()

    # Detect expected frequency
    diffs = df.index.to_series().diff().dropna()
    freq = diffs.mode().iloc[0] if len(diffs) > 0 else pd.Timedelta('15min')

    # Create full index
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    n_gaps = len(full_idx) - len(df)

    if n_gaps > 0:
        df = df.reindex(full_idx)
        # Interpolate numeric columns only (limit=4 -> max 1 hour gap)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].interpolate(method='linear', limit=4, limit_direction='backward')
        # Drop rows that couldn't be interpolated
        df = df.dropna(subset=['DC_POWER', 'IRRADIATION'])

    df = df.reset_index().rename(columns={'index': 'DATE_TIME'})
    return df, n_gaps, freq


def filter_daytime(df):
    """Keep only daytime rows where solar generation is happening."""
    n_before = len(df)
    mask = (
        (df['DC_POWER'] > 0) &
        (df['IRRADIATION'].notna()) &
        (df['IRRADIATION'] > 0) &
        (df['TEMPERATURE'].notna())
    )
    df = df[mask].copy().reset_index(drop=True)
    return df, n_before


def temporal_split(df, train_frac, val_frac):
    """
    Add SPLIT column: train/val/test based on chronological order.
    Strictly temporal  -  no shuffling.
    """
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * (train_frac + val_frac))

    df['SPLIT'] = 'test'
    df.loc[:n_train - 1, 'SPLIT'] = 'train'
    df.loc[n_train:n_val - 1, 'SPLIT'] = 'val'

    return df


def create_forecast_targets(df, horizons):
    """
    Create TARGET_H{h} columns = IRRADIATION shifted forward by h steps.

    At time T, TARGET_H4 = IRRADIATION at time T+4 (1 hour ahead).
    Rows where the target is NaN (end of series) are kept but will be
    excluded during training.
    """
    for h in horizons:
        col_name = f'TARGET_H{h}'
        df[col_name] = df['IRRADIATION'].shift(-h)

    return df


def extract_station_index(filename):
    """Extract station number from filename like 'Solar station site 5 (...).xlsx'"""
    import re
    match = re.search(r'site\s*(\d+)', filename, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def extract_capacity(filename):
    """Extract nominal capacity from filename like '(Nominal capacity-110MW).xlsx'"""
    import re
    match = re.search(r'capacity[- ]*(\d+)\s*MW', filename, re.IGNORECASE)
    return int(match.group(1)) if match else 0


# ---
#  PLOTTING
# ---

def plot_preprocessing(df, station_name, station_idx):
    """Generate preprocessing summary plots for one station."""
    setup_style()

    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    fig.suptitle(f'Station {station_idx}: {station_name}  -  Preprocessed',
                 fontsize=14, fontweight='bold', color=COLORS['primary'])

    # Power
    axes[0].plot(df['DATE_TIME'], df['DC_POWER'], color=COLORS['accent3'],
                 linewidth=0.2, alpha=0.8)
    axes[0].set_ylabel('DC Power (kW)')
    axes[0].grid(True, alpha=0.2)

    # Irradiation
    axes[1].plot(df['DATE_TIME'], df['IRRADIATION'], color=COLORS['accent4'],
                 linewidth=0.2, alpha=0.8)
    axes[1].set_ylabel('Irradiation (kW/m^2)')
    axes[1].grid(True, alpha=0.2)

    # Split visualization
    for split, color in [('train', COLORS['accent3']), ('val', COLORS['accent4']), ('test', COLORS['accent1'])]:
        mask = df['SPLIT'] == split
        if mask.any():
            axes[2].scatter(df.loc[mask, 'DATE_TIME'], df.loc[mask, 'IRRADIATION'],
                          s=0.3, alpha=0.3, c=color, label=f'{split} ({mask.sum():,})')
    axes[2].set_ylabel('Irradiation')
    axes[2].set_xlabel('Date')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.2)

    plt.tight_layout()
    save_figure(fig, f'station_{station_idx:02d}_preprocessing', stage='stage0')

    # Distribution comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Station {station_idx}: Train/Val/Test Distributions',
                 fontsize=12, fontweight='bold', color=COLORS['primary'])

    for ax, col, label in zip(axes, ['DC_POWER', 'IRRADIATION', 'TEMPERATURE'],
                               ['DC Power (kW)', 'Irradiation (kW/m^2)', 'Temperature ( degC)']):
        for split, color in [('train', COLORS['accent3']), ('val', COLORS['accent4']), ('test', COLORS['accent1'])]:
            data = df.loc[df['SPLIT'] == split, col].dropna()
            if len(data) > 0:
                ax.hist(data, bins=50, alpha=0.4, color=color, label=split, density=True)
        ax.set_xlabel(label)
        ax.set_ylabel('Density')
        ax.legend(fontsize=7)

    plt.tight_layout()
    save_figure(fig, f'station_{station_idx:02d}_distributions', stage='stage0')


# ---
#  MAIN ORCHESTRATION
# ---

def run(station_idx=None, manifest=None):
    """
    Main entry point for Stage 0.
    Processes all (or one) Chinese solar stations.
    """
    seed = set_all_seeds()

    raw_dir = get_path("data_raw")
    proc_dir = get_path("data_processed")

    station_cfg = cfg["dataset"]["chinese_stations"]
    pattern = station_cfg["file_pattern"]
    horizons = cfg["forecasting"]["horizons"]
    train_frac = cfg["split"]["train"]
    val_frac = cfg["split"]["val"]

    # Discover files
    files = sorted(glob.glob(str(raw_dir / pattern)))
    if not files:
        # Try recursive search
        files = sorted(glob.glob(str(raw_dir / "**" / pattern), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' in {raw_dir}\n"
            f"Contents: {os.listdir(raw_dir) if raw_dir.exists() else 'DIR NOT FOUND'}"
        )

    # Filter to specific station if requested
    if station_idx is not None:
        files = [f for f in files if f"site {station_idx}" in f.lower() or f"site{station_idx}" in f.lower()]
        if not files:
            raise FileNotFoundError(f"No file found for station {station_idx}")

    print(f"\n  Found {len(files)} station file(s)")

    station_summaries = []

    for filepath in files:
        filepath_obj = Path(filepath)  # Path object
        fname = os.path.basename(filepath)
        sidx = extract_station_index(fname)
        capacity = extract_capacity(fname)

        print(f"\n{'-'*70}")
        print(f"  Station {sidx} ({capacity} MW): {fname}")
        print(f"{'-'*70}")

        if manifest:
            manifest.log_input(fname, filepath_obj)

        t0 = time.time()

        # Step 1: Load
        print(f"  Loading...")
        df = load_station(filepath, station_cfg)
        df_raw = df.copy()   # keep raw copy for sentinel QC
        print(f"    Raw: {df.shape[0]:,} rows x {df.shape[1]} cols")

        # Step 2: Parse timestamps
        df, n_ts_failed = parse_timestamps(df)
        print(f"    Timestamps parsed ({n_ts_failed} failed)")
        print(f"    Date range: {df['DATE_TIME'].min()} -> {df['DATE_TIME'].max()}")

        # Step 3: Clean sentinels
        sentinels = station_cfg.get('sentinels', [-99, -9999, 'null', 'NULL'])
        df, n_cleaned = clean_sentinels(df, sentinels)
        print(f"    Sentinels cleaned: {n_cleaned} values replaced")

        # Step 4: Convert units
        df, unit_info = convert_units(df)
        for k, v in unit_info.items():
            print(f"    {k}: {v}")

        # Step 5: Time features
        df = add_time_features(df)

        # Step 7: Daytime filter
        df, n_before = filter_daytime(df)
        print(f"    Daytime rows: {len(df):,} / {n_before:,} ({len(df)/max(n_before,1)*100:.1f}%)")

        # Pre-hoc QC: flag stations with >30% sentinel values
        sentinel_count = ((df_raw == -99.9) | (df_raw == 9999) | (df_raw == -9999)).sum().sum()
        total_cells = df_raw.shape[0] * df_raw.shape[1]
        sentinel_pct = sentinel_count / total_cells * 100
        if sentinel_pct > 30:
            print(f"    WARNING: Station has {sentinel_pct:.1f}% sentinel values  -  flagged for exclusion")
            df['QC_EXCLUDED'] = True
        else:
            df['QC_EXCLUDED'] = False

        if len(df) < 500:
            print(f"    FAIL Too few rows ({len(df)})  -  SKIPPING")
            continue



        # Step 8: Temporal split (70/15/15)
        df = temporal_split(df, train_frac, val_frac)
        n_train = (df['SPLIT'] == 'train').sum()
        n_val = (df['SPLIT'] == 'val').sum()
        n_test = (df['SPLIT'] == 'test').sum()
        print(f"    Split: train={n_train:,} | val={n_val:,} | test={n_test:,}")

        # Step 9: Create forecast targets
        df = create_forecast_targets(df, horizons)
        target_cols = [f'TARGET_H{h}' for h in horizons]
        n_valid_targets = {h: df[f'TARGET_H{h}'].notna().sum() for h in horizons}
        print(f"    Forecast targets: {', '.join(f'H{h}' for h in horizons)}")
        print(f"    Valid targets: {', '.join(f'H{h}={n:,}' for h, n in n_valid_targets.items())}")

        # Step 10: Leakage check
        try:
            check_temporal_leakage(df)
            print(f"    OK Temporal leakage check PASSED")
        except SchemaError as e:
            print(f"    FAIL LEAKAGE DETECTED: {e}")
            continue

        # Step 11: Summary stats
        print(f"\n    DC_POWER:    mean={df['DC_POWER'].mean():,.0f} kW, max={df['DC_POWER'].max():,.0f} kW")
        print(f"    IRRADIATION: mean={df['IRRADIATION'].mean():.4f}, max={df['IRRADIATION'].max():.4f} kW/m^2")
        print(f"    TEMPERATURE: mean={df['TEMPERATURE'].mean():.1f} degC, range=[{df['TEMPERATURE'].min():.1f}, {df['TEMPERATURE'].max():.1f}]")

        # Step 12: Save
        out_name = f"station_{sidx:02d}_prepared.csv"
        out_path = proc_dir / out_name
        df.to_csv(out_path, index=False)
        elapsed = time.time() - t0
        print(f"\n    OK Saved: {out_name} ({len(df):,} rows x {len(df.columns)} cols) [{elapsed:.1f}s]")

        if manifest:
            manifest.log_output(out_name, out_path)
            manifest.log_metric(f"station_{sidx}_rows", len(df))
            manifest.log_metric(f"station_{sidx}_train", n_train)
            manifest.log_metric(f"station_{sidx}_val", n_val)
            manifest.log_metric(f"station_{sidx}_test", n_test)

        # Step 13: Plot
        try:
            plot_preprocessing(df, fname, sidx)
        except Exception as e:
            print(f"    ! Plot failed: {e}")

        station_summaries.append({
            'station': sidx, 'capacity_mw': capacity,
            'rows': len(df), 'train': n_train, 'val': n_val, 'test': n_test,
            'pwr_max': df['DC_POWER'].max(), 'irr_max': df['IRRADIATION'].max(),
            'date_start': str(df['DATE_TIME'].min()), 'date_end': str(df['DATE_TIME'].max()),
        })

    # Summary table
    if station_summaries:
        print(f"\n{'-'*70}")
        print(f"  STAGE 0 SUMMARY: {len(station_summaries)} stations processed")
        print(f"{'-'*70}")
        print(f"  {'Station':>8} {'Cap(MW)':>8} {'Rows':>8} {'Train':>8} {'Val':>6} {'Test':>6} {'IRR max':>8}")
        print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*8}")
        for s in station_summaries:
            print(f"  {s['station']:>8} {s['capacity_mw']:>8} {s['rows']:>8,} {s['train']:>8,} {s['val']:>6,} {s['test']:>6,} {s['irr_max']:>8.4f}")

        total_rows = sum(s['rows'] for s in station_summaries)
        print(f"\n  Total rows: {total_rows:,}")
        print(f"  Horizons: {horizons}")
        print(f"  Split: {train_frac:.0%} / {val_frac:.0%} / {1-train_frac-val_frac:.0%}")

    # Save summary metrics
    summary_metrics = {
        'n_stations': len(station_summaries),
        'total_rows': sum(s['rows'] for s in station_summaries),
        'stations': station_summaries,
        'horizons': horizons,
        'split_ratios': {'train': train_frac, 'val': val_frac, 'test': 1 - train_frac - val_frac},
    }
    save_metrics(summary_metrics, 'stage0', get_path('outputs_artifacts'))

    print(f"\n  - Stage 0 complete")
    return station_summaries


# ---
#  STANDALONE EXECUTION
# ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 0: Preprocessing")
    parser.add_argument("--station", type=int, default=None,
                       help="Process specific station (1-8)")
    args = parser.parse_args()

    from utils.config import manifest as create_manifest
    m = create_manifest("stage0_preprocessing")

    try:
        run(station_idx=args.station, manifest=m)
    except Exception as e:
        m.log_param("error", str(e))
        raise
    finally:
        manifest_path = m.save()
        print(f"  OK Manifest: {manifest_path}")
