#!/usr/bin/env python3
"""
Watch for the user-attached dataset and import it into data/raw, then
trigger a retrain.

Looks in these possible locations (in order):
  /home/user/uploads/flood_risk_dataset_india.csv
  ./data/raw/flood_risk_dataset_india.csv
  ~/Downloads/flood_risk_dataset_india.csv

Run once:
  python scripts/watch_and_import.py

Or run in a loop (polls every 5s):
  python scripts/watch_and_import.py --watch
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402

CANDIDATE_PATHS = [
    Path("/home/user/uploads/flood_risk_dataset_india.csv"),
    Path.home() / "Downloads" / "flood_risk_dataset_india.csv",
    Path("/tmp/flood_risk_dataset_india.csv"),
]


def find_dataset() -> Path | None:
    # 1. explicit candidates
    for p in CANDIDATE_PATHS:
        if p.exists() and p.stat().st_size > 10_000:
            return p
    # 2. anywhere in data/raw (already imported)
    target = Config.RAW_DATA_DIR / "flood_risk_dataset_india.csv"
    if target.exists() and target.stat().st_size > 10_000:
        return target
    # 3. any flood csv in data/raw
    if Config.RAW_DATA_DIR.exists():
        for f in Config.RAW_DATA_DIR.glob("*.csv"):
            if "flood" in f.name.lower() and f.stat().st_size > 10_000:
                return f
    return None


def validate_csv(path: Path) -> tuple[bool, str]:
    """Basic sanity: header must contain known columns."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            header = fh.readline().lower()
        required = ["rainfall", "water"]
        missing = [c for c in required if c not in header]
        if missing:
            return False, f"Missing expected columns: {missing}"
        size_kb = path.stat().st_size / 1024
        return True, f"OK ({size_kb:.1f} KB)"
    except Exception as e:
        return False, str(e)


def import_and_train() -> bool:
    Config.ensure_dirs()
    src = find_dataset()
    if src is None:
        return False
    ok, msg = validate_csv(src)
    if not ok:
        print(f"[!] Found file {src} but validation failed: {msg}")
        return False

    dest = Config.RAW_DATA_DIR / "flood_risk_dataset_india.csv"
    if src.resolve() != dest.resolve():
        shutil.copy(src, dest)
        print(f"[+] Imported dataset: {src} → {dest}")
    else:
        print(f"[+] Dataset already in place: {dest}")

    print("[*] Starting training on the REAL dataset...")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_pipeline.py")],
        cwd=str(ROOT),
    )
    return result.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true",
                    help="Keep polling until a dataset appears, then import+train.")
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    if not args.watch:
        found = find_dataset()
        if found is None:
            print("No dataset found yet. Place flood_risk_dataset_india.csv at "
                  "/home/user/uploads/ or data/raw/, or re-run with --watch.")
            return 1
        if import_and_train():
            print("[✓] Training completed. Restart the Flask app to pick up the new model.")
            return 0
        return 2

    print(f"[i] Watching for dataset every {args.interval}s (Ctrl+C to stop)...")
    while True:
        if find_dataset() is not None:
            if import_and_train():
                print("[✓] Done. Restart the Flask app to use the real-data model.")
                return 0
            else:
                print("[!] Import/train failed; continuing to watch...")
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
