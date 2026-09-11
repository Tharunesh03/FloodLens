"""
Train the model if no trained artifacts exist. Idempotent.

Used in local setup and on free-tier deployment to guarantee the app has a
model to serve. By default trains on the real dataset if present, otherwise
falls back to synthetic data (labelled as such in the UI).
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402


def main(force_synthetic: bool = False) -> None:
    Config.ensure_dirs()
    model_path = Config.MODELS_DIR / Config.MODEL_FILE
    meta_path = Config.MODELS_DIR / Config.METADATA_FILE
    if model_path.exists() and meta_path.exists():
        logging.info("Model artifacts already exist at %s; skipping training.", model_path)
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("No trained model found. Running training pipeline (force_synthetic=%s)...", force_synthetic)
    from src.train import run_training  # noqa: E402
    meta = run_training(force_synthetic=force_synthetic)
    logging.info("Training complete: %s (f1_macro=%.3f)",
                 meta["model_selection"]["selected_model"],
                 meta["test_metrics"]["f1_macro"])


if __name__ == "__main__":
    force = "--synthetic" in sys.argv
    main(force_synthetic=force)
