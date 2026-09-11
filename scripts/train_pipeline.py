"""Convenience script to run the full training pipeline."""
import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train import run_training  # noqa: E402


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    force_synth = "--synthetic" in sys.argv
    meta = run_training(force_synthetic=force_synth)
    print("\n=== Training complete ===")
    print(f"Selected model: {meta['model_selection']['selected_model']}")
    print(f"Is synthetic data: {meta['is_synthetic_data']}")
    print("Test metrics:")
    for k, v in meta["test_metrics"].items():
        print(f"  {k:>16s}: {v:.4f}")
