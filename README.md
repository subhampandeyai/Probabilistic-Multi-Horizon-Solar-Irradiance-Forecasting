# Multi-Band Heterogeneous Stacked Ensemble for Solar Irradiance Forecasting

A multi-horizon global horizontal irradiance (GHI) forecasting method that
decomposes the signal by frequency band, fits a different learner to each band,
and fuses the component forecasts with a regularised linear meta-learner.
Prediction intervals are produced by split conformal calibration.

## Method

A level-3 Daubechies-4 discrete wavelet transform separates the irradiance
series into four additive components:

| Band | Content | Learner |
|---|---|---|
| `cA3` | slowly varying daily envelope | Ridge |
| `cD3` | diurnal variation | LSTM |
| `cD2` | intra-hour variability | XGBoost |
| `cD1` | high-frequency residual | persistence |

Each learner receives the same shared feature matrix — lags, trailing rolling
statistics, temporal differences, sinusoidal calendar encodings, meteorological
covariates and per-band derivatives — so any difference between component
forecasts reflects the learner rather than unequal inputs.

The four component forecasts are combined by an L2-regularised Ridge
meta-learner:

```
ŷ = β₀ + β₁ŷ_cA3 + β₂ŷ_cD3 + β₃ŷ_cD2 + β₄ŷ_cD1
```

The meta-learner is fitted on validation-split predictions, not on the samples
used to fit the base learners, which keeps the two levels separate.

Split conformal calibration wraps the point forecast in a distribution-free
interval: absolute residuals on a calibration split act as nonconformity
scores, and the interval is `[ŷ − q_α, ŷ + q_α]` where `q_α` is the
`⌈(n+1)(1−α)⌉`-th order statistic. Coverage is finite-sample and assumes no
residual distribution.

## Install

```bash
pip install -r requirements.txt
```

Python 3.9. TensorFlow is required for the LSTM band; on Windows install the
CPU wheel (`pip install tensorflow-cpu==2.20.0`). Training stops rather than
substituting another learner for that band — a substitution makes the daily and
hourly bands fit the same model on the same features, and their meta-learner
coefficients then carry no information.

## Input data

The pipeline expects a time series of irradiance with optional meteorological
covariates. Column names are resolved by keyword from `src/config.yaml`, so
source files do not have to follow a fixed schema:

```yaml
dataset:
  chinese_stations:
    file_pattern: "Solar station site *.xlsx"
    time_col: "Time(year-month-day h:m:s)"
    irr_col_keywords: ["total solar"]
    temp_col_keywords: ["air temp"]
    sentinels: [-99, -9999, "null", "NULL", "--"]
```

Place source files in `data/raw/`. Stage 0 writes cleaned, split and
target-augmented tables to `data/processed/`.

No data is distributed with this repository. The datasets used during
development are public: the Chinese State Grid renewable-energy forecasting
competition dataset (Chen and Xu, *Scientific Data* 9:577, 2022) and the Kaggle
Solar Power Generation Dataset, the latter fetchable with
`python scripts/download_data.py`.

## Usage

Run the stages in order; each reads the previous stage's output.

```bash
python scripts/run_pipeline.py --stage 0,1,2,3   # preprocess -> train
python scripts/run_pipeline.py --stage 0         # a single stage
python scripts/run_pipeline.py --stage 3 --station 5
python scripts/run_pipeline.py --stage all --dry-run
```

| Stage | Module | Produces |
|---|---|---|
| 0 | `src/pipeline/stage0_preprocessing.py` | cleaned series, splits, forecast targets |
| 1 | `src/pipeline/stage1_decomposition.py` | frequency-band components |
| 2 | `src/pipeline/stage2_feature_engineering.py` | shared feature matrix |
| 3 | `src/pipeline/stage3_model_training.py` | per-band models, fused forecast, meta-weights |
| 4 | `src/pipeline/stage4_evaluation.py` | cross-station summary and significance tests |
| 5 | `src/pipeline/stage5_cross_validation.py` | evaluation on an independent dataset |

Reference models and analysis:

```bash
python src/baselines/train_lightgbm_baseline.py
python src/baselines/train_deep_baselines.py      # resumable
python src/analysis/model_comparison.py           # mean R2 per model and horizon
python src/analysis/statistics.py                 # paired tests, effect sizes
python src/analysis/meta_coefficients.py          # band contribution by horizon
python src/conformal_prediction.py                # interval coverage and width
python scripts/export_sample_predictions.py
python scripts/make_figures.py
```

Outputs are written to `results/` and `figures/`; both start empty.

## API

The decomposition and metric helpers are importable directly:

```python
from pipeline.stage1_decomposition import decompose_wavelet, decompose_causal
from utils.metrics import compute_all, bootstrap_ci

components, recon_error = decompose_wavelet(signal, family="db4", level=3)
components, recon_error = decompose_causal(signal, window=512)

metrics = compute_all(y_true, y_pred)      # r2, rmse, mae, mape
low, high = bootstrap_ci(y_true, y_pred)
```

`decompose_wavelet` transforms the whole series at once. `decompose_causal`
reconstructs each band from past samples only, so the value at time *t* never
depends on samples after *t*; it costs O(n·window) instead of O(n).

## Configuration

`src/config.yaml` is the single source of truth: paths, seeds, the 70/15/15
chronological split, forecast horizons, decomposition settings, the band-to-
learner map and every model hyperparameter. Modules resolve paths through
`utils/config.py`; none are hardcoded. Scripts locate the repository root from
their own file location and run from any working directory.

Each stage writes a JSON run manifest recording inputs, outputs, parameters and
file hashes.

## Method notes

The default decomposition is applied to the full series. No parameters are
fitted from the data, so no model state crosses the split, but the wavelet
transform is not a causal filter: a reconstructed band at time *t* draws on
samples on both sides of *t*. Use `decompose_causal()` where a strictly causal
representation is required.

Feature construction is backward-looking throughout: rolling windows are
trailing and lags use positive shifts.

## License

MIT. See `LICENSE`.
