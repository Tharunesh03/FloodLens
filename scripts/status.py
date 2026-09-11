#!/usr/bin/env python3
"""
Quick status check: shows if the real CC0 dataset is present and whether the
model has been trained on it. Useful for debugging and demos.
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import Config, real_dataset_status  # noqa: E402


def main():
    Config.ensure_dirs()
    ds = real_dataset_status()
    print("=" * 60)
    print("FloodLens — Dataset & Model Status")
    print("=" * 60)
    print(f"Real dataset present : {'YES' if ds['available'] else 'NO'}")
    if ds['available']:
        print(f"  File               : {ds.get('name')}")
        print(f"  Size               : {ds.get('size_kb')} KB")
        print(f"  Path               : {ds.get('path')}")
    else:
        print(f"  Expected at        : {ds['expected_path']}")
        print(f"  Upload location    : /home/user/uploads/flood_risk_dataset_india.csv")
        print(f"  Import command     : python scripts/watch_and_import.py")

    meta_path = Config.MODELS_DIR / Config.METADATA_FILE
    if meta_path.exists():
        with open(meta_path) as f:
            m = json.load(f)
        print(f"Model trained        : YES")
        print(f"  Version            : {m['model_version']}")
        print(f"  Algorithm          : {m['model_selection']['selected_model']}")
        print(f"  Trained on         : {'REAL CC0 data' if not m['is_synthetic_data'] else 'SYNTHETIC (demo)'}")
        print(f"  Test F1 (macro)    : {m['test_metrics']['f1_macro']:.3f}")
        print(f"  Test Recall (macro): {m['test_metrics']['recall_macro']:.3f}")
    else:
        print("Model trained        : NO")
        print("  Train command      : python scripts/train_pipeline.py")

    # Check for uploads waiting
    up = Path("/home/user/uploads/flood_risk_dataset_india.csv")
    if up.exists() and up.stat().st_size > 10000:
        print("Upload detected      : YES (will be picked up by watch_and_import.py)")


if __name__ == "__main__":
    main()
