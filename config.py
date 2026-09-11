"""
FloodLens configuration.
Reads from environment variables with sensible defaults for local development.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Load .env if present (never crash if missing)
load_dotenv(BASE_DIR / ".env")


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    DEBUG = os.getenv("FLASK_DEBUG", "1") == "1"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5000"))

    # Paths
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
    MODELS_DIR = BASE_DIR / "models"
    STATIC_DIR = BASE_DIR / "app" / "static"
    EXPORTS_DIR = STATIC_DIR / "exports"
    INSTANCE_DIR = BASE_DIR / "instance"

    # Database
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{INSTANCE_DIR / 'floodlens.db'}",
    )

    # Model metadata
    MODEL_VERSION = "1.0.0"
    MODEL_FILE = "floodlens_model.joblib"
    PREPROCESSOR_FILE = "floodlens_preprocessor.joblib"
    METADATA_FILE = "floodlens_metadata.json"

    # Risk levels
    RISK_LEVELS = {
        0: "Low",
        1: "Moderate",
        2: "High",
        3: "Critical",
    }
    RISK_COLORS = {
        "Low": "#10b981",        # green
        "Moderate": "#f59e0b",   # amber
        "High": "#ef4444",       # red
        "Critical": "#b91c1c",   # deep red
    }

    # Recognized real dataset
    REAL_DATASET_FILENAME = "flood_risk_india.csv"
    SYNTHETIC_DATASET_FILENAME = "flood_risk_synthetic.csv"

    # Geographic coverage (for disclosure)
    GEO_COVERAGE = "India (synthetic fallback used when real dataset is unavailable)"

    @classmethod
    def ensure_dirs(cls):
        for p in [
            cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR,
            cls.SYNTHETIC_DATA_DIR, cls.MODELS_DIR, cls.EXPORTS_DIR,
            cls.INSTANCE_DIR,
        ]:
            p.mkdir(parents=True, exist_ok=True)


# Module-level aliases so callers can `from config import RISK_LEVELS`
RISK_LEVELS = Config.RISK_LEVELS
RISK_COLORS = Config.RISK_COLORS


def real_dataset_status() -> dict:
    """Return a dict describing whether the real CC0 dataset is present."""
    import os
    from src.data_processing import _try_find_real_dataset
    p = _try_find_real_dataset()
    if p is not None:
        size_kb = p.stat().st_size / 1024
        return {
            "available": True,
            "name": p.name,
            "size_kb": round(size_kb, 1),
            "path": str(p.relative_to(BASE_DIR)) if str(p).startswith(str(BASE_DIR)) else str(p),
        }
    return {
        "available": False,
        "expected_path": str((Config.RAW_DATA_DIR / "flood_risk_dataset_india.csv").relative_to(BASE_DIR)),
    }
