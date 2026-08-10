"""
Download the datasets needed to reproduce the experiments.

Datasets:
  1. Primary: 8 Chinese State Grid solar stations (mirror of Chen & Xu, Sci. Data, 2022)
     Original source: https://doi.org/10.1038/s41597-022-01696-6 (Chen & Xu, 2022)
     Hosted mirror:   https://www.kaggle.com/datasets/kagglesubham/multi-site-wind-and-solar-power-generation-dataset
     Files placed in: data/raw/

  2. External: 2 Indian PV plants (Kannal, Kaggle, 2020)
     URL: https://www.kaggle.com/datasets/anikannal/solar-power-generation-data
     Files placed in: data/external/

Both datasets are fetched automatically via kagglehub.
"""
import argparse
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_EXT = ROOT / "data" / "external"


def download_primary():
    """Chinese State Grid stations (mirror of Sci. Data 2022 dataset)."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Primary dataset: 8 Chinese State Grid stations")
    print("Original source: Chen & Xu, Sci. Data, 2022 (DOI: 10.1038/s41597-022-01696-6)")
    print("Hosted mirror:   kagglesubham/multi-site-wind-and-solar-power-generation-dataset")
    print("=" * 70)

    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub is not installed.")
        print("Install with:  pip install kagglehub")
        sys.exit(1)

    path = kagglehub.dataset_download(
        "kagglesubham/multi-site-wind-and-solar-power-generation-dataset"
    )
    print(f"Downloaded to: {path}")

    src = Path(path)
    copied = 0
    for xlsx in src.rglob("Solar station site*.xlsx"):
        dest = DATA_RAW / xlsx.name
        shutil.copy2(xlsx, dest)
        copied += 1
        print(f"  copied: {xlsx.name}")

    if copied != 8:
        print(f"WARNING: copied {copied} solar station files, expected 8.")
    print(f"Files placed in: {DATA_RAW}")


def download_external():
    """Indian PV plants dataset from Kaggle."""
    DATA_EXT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("External dataset: 2 Indian PV plants (Kannal, Kaggle, 2020)")
    print("URL: https://www.kaggle.com/datasets/anikannal/solar-power-generation-data")
    print("=" * 70)

    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub is not installed.")
        print("Install with:  pip install kagglehub")
        sys.exit(1)

    path = kagglehub.dataset_download("anikannal/solar-power-generation-data")
    print(f"Downloaded to: {path}")

    expected = [
        "Plant_1_Generation_Data.csv",
        "Plant_1_Weather_Sensor_Data.csv",
        "Plant_2_Generation_Data.csv",
        "Plant_2_Weather_Sensor_Data.csv",
    ]
    src = Path(path)
    for fname in expected:
        s = src / fname
        d = DATA_EXT / fname
        if s.exists():
            shutil.copy2(s, d)
            print(f"  copied: {fname}")
        else:
            print(f"  WARNING: {fname} not found at {s}")

    print(f"Files placed in: {DATA_EXT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-primary", action="store_true",
                        help="Skip the Chinese State Grid dataset")
    parser.add_argument("--skip-external", action="store_true",
                        help="Skip the Kaggle Indian plants dataset")
    args = parser.parse_args()

    if not args.skip_primary:
        download_primary()
        print()
    if not args.skip_external:
        download_external()
        print()

    print("Done. Verify file presence:")
    print("    Get-ChildItem data\\raw")
    print("    Get-ChildItem data\\external")


if __name__ == "__main__":
    main()