"""
Pipeline runner for the multi-band heterogeneous stacking framework.

Executes the stages in dependency order, validates the schema before and after
each one, and writes a run manifest recording inputs, outputs and parameters.
All paths are resolved from config.yaml.

    python scripts/run_pipeline.py --stage all              # full pipeline
    python scripts/run_pipeline.py --stage 0                # single stage
    python scripts/run_pipeline.py --stage 0,1,2            # selected stages
    python scripts/run_pipeline.py --stage 3 --station 5    # single station
    python scripts/run_pipeline.py --stage all --dry-run    # validate only
    python scripts/run_pipeline.py --verify                 # check outputs
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, sys, time, json
from pathlib import Path
from datetime import datetime

# - Stage registry -
STAGES = {
    0: {
        "name": "stage0_preprocessing",
        "description": "Load, clean, convert units, temporal split, create forecast targets",
        "module": "pipeline.stage0_preprocessing",
        "function": "run",
        "inputs": ["config.yaml", "data/raw/*.xlsx"],
        "outputs": ["data/processed/station_{idx}_prepared.csv"],
    },
    1: {
        "name": "stage1_decomposition",
        "description": "Wavelet/EMD/VMD decomposition of irradiation signal",
        "module": "pipeline.stage1_decomposition",
        "function": "run",
        "inputs": ["data/processed/station_{idx}_prepared.csv"],
        "outputs": ["data/processed/station_{idx}_decomposed.csv"],
    },
    2: {
        "name": "stage2_feature_engineering",
        "description": "Component-specific feature engineering per frequency band",
        "module": "pipeline.stage2_feature_engineering",
        "function": "run",
        "inputs": ["data/processed/station_{idx}_decomposed.csv"],
        "outputs": ["data/processed/station_{idx}_features.csv"],
    },
    3: {
        "name": "stage3_model_training",
        "description": "Frequency-adaptive model training (THE NOVEL PART)",
        "module": "pipeline.stage3_model_training",
        "function": "run",
        "inputs": ["data/processed/station_{idx}_features.csv"],
        "outputs": [
            "outputs/models/station_{idx}_component_models.pkl",
            "outputs/artifacts/stage3_metrics.json",
        ],
    },
    4: {
        "name": "stage4_evaluation",
        "description": "Reconstruct forecast from component predictions + bias correction",
        "module": "pipeline.stage4_evaluation",
        "function": "run",
        "inputs": ["outputs/models/station_{idx}_component_models.pkl"],
        "outputs": [
            "data/processed/station_{idx}_predictions.csv",
            "outputs/artifacts/stage4_metrics.json",
        ],
    },
    5: {
        "name": "stage5_cross_validation",
        "description": "Multi-station, multi-horizon evaluation + statistical tests",
        "module": "pipeline.stage5_cross_validation",
        "function": "run",
        "inputs": ["data/processed/station_*_predictions.csv"],
        "outputs": [
            "outputs/artifacts/stage5_metrics.json",
            "results/evaluation_report.txt",
        ],
    },
    6: {
        "name": "stage6_ablation",
        "description": "Ablation study: uniform model, no decomposition, level sweep, window sweep",
        "module": "pipeline.stage6_ablation",
        "function": "run",
        "inputs": ["data/processed/station_*_features.csv"],
        "outputs": [
            "outputs/artifacts/stage6_ablation_metrics.json",
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="FAME Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --stage all
  python run_pipeline.py --stage 0,1,2
  python run_pipeline.py --stage 3 --station 5
  python run_pipeline.py --verify
        """
    )
    parser.add_argument("--stage", type=str, default="all",
                       help="Stage(s) to run: 'all', single number, or comma-separated")
    parser.add_argument("--station", type=int, default=None,
                       help="Process specific station index (1-8)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Validate config and schemas only, don't execute")
    parser.add_argument("--verify", action="store_true",
                       help="Verify all artifacts exist and are consistent")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to config file")
    return parser.parse_args()


def resolve_stages(stage_arg: str) -> list:
    """Parse stage argument into list of stage numbers."""
    if stage_arg == "all":
        return sorted(STAGES.keys())
    return sorted(int(s.strip()) for s in stage_arg.split(","))


def verify_artifacts():
    """Verify all artifacts are consistent."""
    from utils.config import get_path
    artifact_dir = get_path("outputs_artifacts")

    print("\n  Verifying pipeline artifacts...")
    all_ok = True

    for stage_num in sorted(STAGES.keys()):
        stage = STAGES[stage_num]
        manifest_path = artifact_dir / f"{stage['name']}_latest.json"

        if manifest_path.exists():
            with open(manifest_path) as f:
                m = json.load(f)

            # Check all output files still exist
            outputs_ok = True
            for name, info in m.get("output_files", {}).items():
                p = Path(info["path"])
                if not p.exists():
                    print(f"    FAIL Stage {stage_num}: Output missing: {name} -> {p}")
                    outputs_ok = False
                    all_ok = False

            if outputs_ok:
                ts = m.get("timestamp_end", "?")
                n_metrics = len(m.get("metrics", {}))
                print(f"    OK Stage {stage_num} ({stage['name']}): OK | {ts} | {n_metrics} metrics")
        else:
            print(f"    - Stage {stage_num} ({stage['name']}): Not yet run")

    if all_ok:
        print("\n  OK All artifacts consistent.")
    else:
        print("\n  ! Some artifacts missing. Re-run affected stages.")
    return all_ok


def run_stage(stage_num: int, station: int = None, dry_run: bool = False):
    """Execute a single stage with full provenance tracking."""
    import importlib
    from utils.config import manifest, set_all_seeds

    stage = STAGES[stage_num]
    print(f"\n{'-'*70}")
    print(f"  Stage {stage_num}: {stage['description']}")
    print(f"{'-'*70}")

    if dry_run:
        print(f"  [DRY RUN] Would execute: {stage['module']}.{stage['function']}()")
        return True

    # Create manifest
    m = manifest(stage["name"])
    m.log_param("station", station)

    # Set seeds
    seed = set_all_seeds()
    m.log_param("seed", seed)

    # Import and run
    t0 = time.time()
    try:
        module = importlib.import_module(stage["module"])
        run_fn = getattr(module, stage["function"])

        # Pass station arg if provided
        if station is not None:
            result = run_fn(station_idx=station, manifest=m)
        else:
            result = run_fn(manifest=m)

        elapsed = time.time() - t0
        m.log_param("elapsed_seconds", round(elapsed, 1))
        manifest_path = m.save()

        print(f"\n  OK Stage {stage_num} completed in {elapsed:.1f}s")
        print(f"  OK Manifest: {manifest_path}")
        return True

    except Exception as e:
        elapsed = time.time() - t0
        m.log_param("elapsed_seconds", round(elapsed, 1))
        m.log_param("error", str(e))
        m.save()
        print(f"\n  FAIL Stage {stage_num} FAILED after {elapsed:.1f}s: {e}")
        raise


def main():
    args = parse_args()

    print("-"*70)
    print("  FAME - Frequency-Adaptive Multi-Resolution Ensemble")
    print("  Solar Irradiance Forecasting Pipeline")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*70)

    # Verify mode
    if args.verify:
        verify_artifacts()
        return

    # Resolve stages
    stages = resolve_stages(args.stage)
    print(f"\n  Stages to run: {stages}")
    if args.station:
        print(f"  Station: {args.station}")
    if args.dry_run:
        print(f"  Mode: DRY RUN (validation only)")

    # Execute
    t_total = time.time()
    for stage_num in stages:
        if stage_num not in STAGES:
            print(f"\n  FAIL Unknown stage: {stage_num}")
            sys.exit(1)
        run_stage(stage_num, station=args.station, dry_run=args.dry_run)

    total = time.time() - t_total
    print(f"\n{'-'*70}")
    print(f"  Pipeline complete. Total time: {total:.1f}s")
    print(f"{'-'*70}")


if __name__ == "__main__":
    main()

