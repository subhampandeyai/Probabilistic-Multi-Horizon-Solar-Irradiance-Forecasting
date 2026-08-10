"""
FAME  -  Metrics Utility
=======================
ALL metrics computed at runtime from data.
              ZERO hardcoded values anywhere.

Usage:
    from utils.metrics import compute_all, load_stage_metrics
    m = compute_all(y_true, y_pred)
    prev = load_stage_metrics("stage1")  # Read from JSON artifact
"""
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all standard regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE (skip zeros to avoid division errors)
    mask = np.abs(y_true) > 1e-8
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan

    # Forecast Skill Score vs persistence (FSS)
    # FSS = 1 - RMSE_model / RMSE_persistence
    # Persistence = y_true shifted by 1 (last known value)
    if len(y_true) > 1:
        rmse_persist = np.sqrt(np.mean((y_true[1:] - y_true[:-1])**2))
        fss = 1 - rmse / rmse_persist if rmse_persist > 0 else np.nan
    else:
        fss = np.nan

    return {
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "fss": float(fss),
        "n_samples": int(len(y_true)),
    }


def bootstrap_ci(y_true, y_pred, metric_fn=r2_score, n_boot=500, ci=95):
    """Bootstrap confidence interval for any metric."""
    np.random.seed(42)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    lo = np.percentile(scores, (100 - ci) / 2)
    hi = np.percentile(scores, 100 - (100 - ci) / 2)
    return float(lo), float(hi)


def save_metrics(metrics: dict, stage_name: str, output_dir: Path):
    """Save metrics to a JSON artifact file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stage_name}_metrics.json"
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    return path


def load_stage_metrics(stage_name: str, artifacts_dir: Path = None) -> dict:
    """
    Load metrics from a saved JSON artifact.
    This is the ONLY way plots/reports get metrics.
    """
    if artifacts_dir is None:
        from utils.config import get_path
        artifacts_dir = get_path("outputs_artifacts")

    path = artifacts_dir / f"{stage_name}_metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics artifact not found: {path}\n"
            f"Run {stage_name} first before generating reports/plots."
        )
    with open(path, 'r') as f:
        return json.load(f)
