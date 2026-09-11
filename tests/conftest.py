"""Pytest fixtures: build a minimal app + tiny trained model for tests."""
from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Use a temporary instance/model directory to avoid touching real artifacts.
os.environ.setdefault("FLASK_ENV", "testing")


@pytest.fixture(scope="session")
def app_with_model(tmp_path_factory):
    """Create an app against a fresh temp dir with a tiny trained model."""
    tmp = tmp_path_factory.mktemp("floodlens_test")
    (tmp / "data" / "raw").mkdir(parents=True)
    (tmp / "data" / "processed").mkdir(parents=True)
    (tmp / "data" / "synthetic").mkdir(parents=True)
    (tmp / "models").mkdir(parents=True)
    (tmp / "instance").mkdir(parents=True)
    (tmp / "app" / "static" / "exports").mkdir(parents=True)

    # Override config paths via monkeypatching Config class attributes
    import config as cfg_mod
    from config import Config

    overrides = {
        "DATA_DIR": tmp / "data",
        "RAW_DATA_DIR": tmp / "data" / "raw",
        "PROCESSED_DATA_DIR": tmp / "data" / "processed",
        "SYNTHETIC_DATA_DIR": tmp / "data" / "synthetic",
        "MODELS_DIR": tmp / "models",
        "STATIC_DIR": ROOT / "app" / "static",
        "EXPORTS_DIR": tmp / "app" / "static" / "exports",
        "INSTANCE_DIR": tmp / "instance",
        "DATABASE_URL": f"sqlite:///{tmp / 'instance' / 'floodlens.db'}",
    }
    for k, v in overrides.items():
        setattr(Config, k, v)
    Config.ensure_dirs()

    # Train a tiny model on a small synthetic dataset for speed
    from src.train import run_training
    # Patch n_samples via temporarily calling generate_synthetic_dataset directly
    from src import data_processing as dp
    original = dp.generate_synthetic_dataset

    def small_synth(n_samples=2000, seed=42):
        return original(n_samples=2000, seed=seed)

    dp.generate_synthetic_dataset = small_synth
    try:
        meta = run_training(force_synthetic=True)
    finally:
        dp.generate_synthetic_dataset = original

    from app import create_app
    app = create_app({
        "TESTING": True,
        "WTF_CSRF_ENABLED": False,
        "SECRET_KEY": "test-secret-key",
    })
    app.config["TEST_DIR"] = tmp
    yield app


@pytest.fixture
def client(app_with_model):
    return app_with_model.test_client()
