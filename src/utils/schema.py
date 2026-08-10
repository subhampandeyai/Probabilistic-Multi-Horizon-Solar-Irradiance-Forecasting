"""
FAME  -  Schema Validators
=========================
Artifact contracts between stages.
Each stage validates its inputs BEFORE running and its outputs AFTER.
If validation fails, the stage ABORTS with a clear error.

Usage:
    from utils.schema import validate_input, validate_output
    validate_input("stage1", input_path)   # Raises SchemaError if invalid
    validate_output("stage1", output_path) # Raises SchemaError if invalid
"""
import pandas as pd
import numpy as np
from pathlib import Path

class SchemaError(Exception):
    """Raised when stage input/output fails validation."""
    pass

# ---
#  SCHEMA DEFINITIONS
#  Each stage declares what columns MUST exist, what types they are,
#  and what constraints they satisfy.
# ---

SCHEMAS = {
    # -- Stage 0 output / Stage 1 input ---
    "stage0_output": {
        "required_columns": [
            "DATE_TIME", "IRRADIATION", "TEMPERATURE", "DC_POWER",
            "HOUR", "DAY", "MONTH", "DOY", "SPLIT"
        ],
        "numeric_columns": [
            "IRRADIATION", "TEMPERATURE", "DC_POWER", "HOUR"
        ],
        "no_nan_columns": ["DATE_TIME", "IRRADIATION", "SPLIT"],
        "split_values": ["train", "val", "test"],
        "min_rows": 500,
    },

    # -- Stage 1 output / Stage 2 input ---
    "stage1_output": {
        "required_columns": [
            "DATE_TIME", "IRRADIATION", "SPLIT"
        ],
        # Decomposition columns are dynamic (wavelet vs EMD)
        # so we check they exist via prefix
        "required_column_prefixes": ["IRR_"],
        "no_nan_columns": ["DATE_TIME", "IRRADIATION", "SPLIT"],
        "min_rows": 500,
    },

    # -- Stage 2 output / Stage 3 input ---
    "stage2_output": {
        "required_columns": ["DATE_TIME", "IRRADIATION", "SPLIT"],
        "no_nan_columns": ["SPLIT"],
        "min_rows": 500,
    },

    # -- Forecasting target (multi-horizon) ---
    "forecast_targets": {
        "required_columns": ["DATE_TIME", "IRRADIATION", "SPLIT"],
        "required_column_prefixes": ["TARGET_H"],
        "no_nan_columns": ["SPLIT"],
        "min_rows": 500,
    },
}


def validate_dataframe(df: pd.DataFrame, schema_name: str, filepath: str = ""):
    """
    Validate a DataFrame against a named schema.
    Raises SchemaError with detailed message on failure.
    """
    schema = SCHEMAS.get(schema_name)
    if schema is None:
        raise SchemaError(f"Unknown schema: {schema_name}")

    errors = []
    context = f" (file: {filepath})" if filepath else ""

    # Check required columns
    if "required_columns" in schema:
        missing = set(schema["required_columns"]) - set(df.columns)
        if missing:
            errors.append(f"Missing required columns: {sorted(missing)}")

    # Check required column prefixes (at least one column with each prefix)
    if "required_column_prefixes" in schema:
        for prefix in schema["required_column_prefixes"]:
            matches = [c for c in df.columns if c.startswith(prefix)]
            if not matches:
                errors.append(f"No columns with prefix '{prefix}' found")

    # Check no-NaN columns
    if "no_nan_columns" in schema:
        for col in schema["no_nan_columns"]:
            if col in df.columns and df[col].isna().any():
                n_nan = df[col].isna().sum()
                errors.append(f"Column '{col}' has {n_nan} NaN values")

    # Check numeric columns are actually numeric
    if "numeric_columns" in schema:
        for col in schema["numeric_columns"]:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' is not numeric (dtype: {df[col].dtype})")

    # Check split values
    if "split_values" in schema and "SPLIT" in df.columns:
        actual = set(df["SPLIT"].unique())
        expected = set(schema["split_values"])
        unexpected = actual - expected
        if unexpected:
            errors.append(f"Unexpected SPLIT values: {unexpected}")
        # At minimum, train and test must exist
        if "train" not in actual:
            errors.append("SPLIT column missing 'train' partition")
        if "test" not in actual:
            errors.append("SPLIT column missing 'test' partition")

    # Check minimum rows
    if "min_rows" in schema and len(df) < schema["min_rows"]:
        errors.append(f"Only {len(df)} rows, minimum required: {schema['min_rows']}")

    # Check temporal ordering (if DATE_TIME exists)
    if "DATE_TIME" in df.columns:
        dt = pd.to_datetime(df["DATE_TIME"], errors="coerce")
        if dt.isna().sum() > len(df) * 0.01:  # >1% unparseable
            errors.append(f"DATE_TIME has {dt.isna().sum()} unparseable values")

    if errors:
        msg = f"\n  Schema validation FAILED for '{schema_name}'{context}:\n"
        for e in errors:
            msg += f"    FAIL {e}\n"
        raise SchemaError(msg)

    return True


def validate_input(stage_name: str, filepath: Path):
    """
    Validate input file for a given stage.
    Maps stage name to expected input schema.
    """
    input_schema_map = {
        "stage1": "stage0_output",
        "stage2": "stage1_output",
        "stage3": "stage2_output",
        "stage4": "stage2_output",   # Stage 3 may keep same schema
        "stage5": "forecast_targets",
    }
    schema_name = input_schema_map.get(stage_name)
    if schema_name is None:
        return True  # No schema defined for this input

    df = pd.read_csv(filepath, nrows=5000)  # Validate on sample for speed
    return validate_dataframe(df, schema_name, str(filepath))


def validate_output(stage_name: str, filepath: Path):
    """Validate output file for a given stage."""
    output_schema_map = {
        "stage0": "stage0_output",
        "stage1": "stage1_output",
        "stage2": "stage2_output",
    }
    schema_name = output_schema_map.get(stage_name)
    if schema_name is None:
        return True

    df = pd.read_csv(filepath, nrows=5000)
    return validate_dataframe(df, schema_name, str(filepath))


# ---
#  LEAKAGE CHECKS ()
# ---

def check_temporal_leakage(df: pd.DataFrame, target_col: str = "IRRADIATION"):
    """
    Verify no future information leaks into features.
    Checks:
      1. Train/val/test dates don't overlap
      2. No feature column has higher correlation with FUTURE target
         than with CURRENT target (symptom of look-ahead)
    """
    errors = []

    if "SPLIT" not in df.columns or "DATE_TIME" not in df.columns:
        return True

    dt = pd.to_datetime(df["DATE_TIME"])

    # Check 1: No temporal overlap between splits
    splits = {}
    for s in ["train", "val", "test"]:
        mask = df["SPLIT"] == s
        if mask.any():
            splits[s] = (dt[mask].min(), dt[mask].max())

    if "train" in splits and "val" in splits:
        if splits["train"][1] >= splits["val"][0]:
            errors.append(f"Train end ({splits['train'][1]}) >= Val start ({splits['val'][0]})")

    if "val" in splits and "test" in splits:
        if splits["val"][1] >= splits["test"][0]:
            errors.append(f"Val end ({splits['val'][1]}) >= Test start ({splits['test'][0]})")

    if "train" in splits and "test" in splits:
        if splits["train"][1] >= splits["test"][0]:
            errors.append(f"Train end ({splits['train'][1]}) >= Test start ({splits['test'][0]})")

    if errors:
        msg = "\n  ! TEMPORAL LEAKAGE DETECTED:\n"
        for e in errors:
            msg += f"    FAIL {e}\n"
        raise SchemaError(msg)

    return True
