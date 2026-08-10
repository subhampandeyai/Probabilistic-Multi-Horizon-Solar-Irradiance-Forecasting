"""
Configuration loading, path resolution and run manifests.

Every module reads its paths and parameters from config.yaml through this
module, so no path is hardcoded and a script behaves the same from any working
directory.

    from utils.config import cfg, get_path, manifest

    data_dir = get_path("data_raw")
    seed = cfg["seeds"]["global"]
"""
import os, sys, yaml, json, hashlib, time, platform
from pathlib import Path
from datetime import datetime

# -- Locate the configuration and the repository root ---
def _find_config_dir() -> Path:
    """Walk upward from this file until config.yaml is found."""
    current = Path(__file__).resolve().parent.parent
    for _ in range(5):
        if (current / "config.yaml").exists():
            return current
        current = current.parent
    raise FileNotFoundError(
        "config.yaml not found. Run from within the repository, or place "
        "config.yaml alongside the package directory."
    )

CONFIG_DIR = _find_config_dir()   # directory holding config.yaml, i.e. src/
REPO_ROOT = CONFIG_DIR.parent     # data/, results/ and figures/ live here

# -- Load config ---
def load_config(config_path: Path = None) -> dict:
    """Load config.yaml and return it as a dict."""
    path = config_path or (CONFIG_DIR / "config.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

cfg = load_config()

# -- Path resolver (always returns absolute paths) ---
def get_path(key: str, create: bool = True) -> Path:
    """
    Resolve a path key from config.yaml to an absolute path.

    Paths in config.yaml are relative to the repository root, which is the
    parent of the directory holding config.yaml. Resolving against the root
    keeps data/, results/ and figures/ in one place regardless of which module
    or working directory the call comes from.

    Creates the directory when create=True and the key names a directory.

        get_path("data_raw") -> <repo>/data/raw
    """
    relative = cfg["paths"].get(key)
    if relative is None:
        raise KeyError(f"Path key '{key}' not found in config.yaml")
    absolute = REPO_ROOT / relative

    if create and not absolute.suffix:  # Only mkdir for directories, not files
        absolute.mkdir(parents=True, exist_ok=True)
    return absolute

# -- Seed management ---
def set_all_seeds(seed: int = None):
    """Seed every random source used by the pipeline.

    Call before any stage that trains a model; the reported results assume the
    seeds in config.yaml.
    """
    import random, numpy as np
    seed = seed or cfg["seeds"]["global"]
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(cfg["seeds"].get("tensorflow", seed))
    except ImportError:
        pass
    return seed

# -- File hashing (for provenance) ---
def file_hash(filepath: Path, algo: str = "sha256") -> str:
    """Compute hash of a file for provenance tracking."""
    h = hashlib.new(algo)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]  # Short hash is sufficient

# -- Run Manifest () ---
class RunManifest:
    """
    Records provenance for each stage run.
    Writes JSON manifest to outputs/artifacts/
    """
    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self.start_time = datetime.now().isoformat()
        self.data = {
            "stage": stage_name,
            "timestamp_start": self.start_time,
            "timestamp_end": None,
            "seed": cfg["seeds"]["global"],
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "config_hash": file_hash(CONFIG_DIR / "config.yaml"),
            "input_files": {},
            "output_files": {},
            "metrics": {},
            "parameters": {},
        }

    def log_input(self, name: str, filepath: Path):
        """Record an input file with its hash."""
        self.data["input_files"][name] = {
            "path": str(filepath),
            "hash": file_hash(filepath) if filepath.exists() else "MISSING",
        }

    def log_output(self, name: str, filepath: Path):
        """Record an output file with its hash."""
        self.data["output_files"][name] = {
            "path": str(filepath),
            "hash": file_hash(filepath) if filepath.exists() else "NOT_YET",
        }

    def log_metric(self, name: str, value):
        """Record a computed metric."""
        self.data["metrics"][name] = value

    def log_param(self, name: str, value):
        """Record a parameter used."""
        self.data["parameters"][name] = value

    def save(self):
        """Write manifest to outputs/artifacts/."""
        self.data["timestamp_end"] = datetime.now().isoformat()
        # Update output file hashes now that files exist
        for name, info in self.data["output_files"].items():
            p = Path(info["path"])
            if p.exists():
                info["hash"] = file_hash(p)

        artifact_dir = get_path("outputs_artifacts")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_path = artifact_dir / f"{self.stage_name}_{ts}.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)

        # Also write "latest" symlink-style file
        latest_path = artifact_dir / f"{self.stage_name}_latest.json"
        with open(latest_path, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)

        return manifest_path

# -- Convenience: create a manifest ---
def manifest(stage_name: str) -> RunManifest:
    return RunManifest(stage_name)
