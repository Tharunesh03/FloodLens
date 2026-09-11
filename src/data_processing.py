"""
Data loading, schema validation, cleaning and synthetic-data generation
for the FloodLens project.

Real dataset preferred: data/raw/flood_risk_india.csv (CC0, Kaggle).
Fallback: synthetic dataset generated at data/synthetic/flood_risk_synthetic.csv,
clearly tagged as synthetic.
"""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Make imports work both as a module and as a script
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import Config  # noqa: E402

logger = logging.getLogger("floodlens.data")

# Columns expected in the real Kaggle dataset (with common variants)
EXPECTED_COLUMNS = [
    "Latitude", "Longitude", "Rainfall_mm", "Temperature_C",
    "Humidity_pct", "River_Discharge_m3_s", "Water_Level_m",
    "Elevation_m", "Land_Cover", "Soil_Type", "Population_Density",
    "Infrastructure", "Historical_Floods", "Flood_Occurred",
]

NUMERIC_COLS = [
    "Latitude", "Longitude", "Rainfall_mm", "Temperature_C",
    "Humidity_pct", "River_Discharge_m3_s", "Water_Level_m",
    "Elevation_m", "Population_Density",
]

CATEGORICAL_COLS = ["Land_Cover", "Soil_Type"]
BINARY_COLS = ["Infrastructure", "Historical_Floods", "Flood_Occurred"]


def _try_find_real_dataset() -> Path | None:
    """Look for the real dataset under data/raw with a few candidate names."""
    candidates = [
        Config.RAW_DATA_DIR / Config.REAL_DATASET_FILENAME,
        Config.RAW_DATA_DIR / "flood_risk_dataset_india.csv",
        Config.RAW_DATA_DIR / "flood risk dataset india.csv",
    ]
    if Config.RAW_DATA_DIR.exists():
        for name in os.listdir(Config.RAW_DATA_DIR):
            if name.lower().endswith(".csv") and "flood" in name.lower():
                candidates.append(Config.RAW_DATA_DIR / name)
    for p in candidates:
        if p and p.exists() and p.is_file() and p.stat().st_size > 10000:
            # Sanity: must contain at least one expected column header
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                    header = fh.readline()
                if "Rainfall" in header or "rainfall" in header or "Flood" in header:
                    return p
            except Exception:
                continue
    return None


def generate_synthetic_dataset(n_samples: int = 50000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset with the same schema as the real
    Kaggle "Flood Risk in India" dataset.

    This is used ONLY when the real dataset is unavailable. Correlations are
    chosen so that the ML pipeline has physically meaningful relationships
    to learn (e.g. high rainfall + high river discharge + low elevation +
    clay/silt soil = elevated flood risk), but no claim is made that these
    numbers represent real measurements.
    """
    rng = np.random.default_rng(seed)

    # Indian lat/lon bounding box (approximate)
    lat = rng.uniform(8.0, 35.0, n_samples)
    lon = rng.uniform(68.0, 97.0, n_samples)

    # Environmental variables with mild correlations
    rainfall = np.clip(rng.gamma(2.0, 30.0, n_samples), 0, 600)
    temperature = rng.normal(26.0, 5.0, n_samples)
    humidity = np.clip(50 + 0.08 * rainfall + rng.normal(0, 8, n_samples), 10, 100)
    river_discharge = np.clip(200 + 3.0 * rainfall + rng.normal(0, 400, n_samples), 50, 12000)
    water_level = np.clip(1.0 + 0.0007 * river_discharge + rng.normal(0, 0.5, n_samples), 0.2, 10)
    elevation = np.clip(rng.lognormal(5.0, 1.0, n_samples), 1, 4000)

    land_cover = rng.choice(
        ["Urban", "Forest", "Agricultural", "Water Body", "Desert"],
        size=n_samples, p=[0.15, 0.25, 0.45, 0.05, 0.10],
    )
    soil_type = rng.choice(
        ["Sandy", "Clay", "Loam", "Silt", "Peat"],
        size=n_samples, p=[0.20, 0.25, 0.30, 0.20, 0.05],
    )
    pop_density = np.clip(rng.lognormal(5.5, 1.2, n_samples), 50, 12000)
    infrastructure = rng.binomial(1, 0.4, n_samples)

    # Historical floods more likely in low-elevation, high-rainfall areas
    hist_p = (
        0.05
        + 0.35 * (rainfall / rainfall.max())
        + 0.25 * (water_level / water_level.max())
        - 0.15 * (elevation / elevation.max())
    )
    hist_p = np.clip(hist_p, 0.02, 0.9)
    historical_floods = rng.binomial(1, hist_p, n_samples)

    # Flood occurrence probability
    soil_penalty = np.where(np.isin(soil_type, ["Clay", "Silt", "Peat"]), 0.10, 0.0)
    cover_penalty = np.where(land_cover == "Water Body", 0.15, 0.0)
    infra_benefit = np.where(infrastructure == 1, -0.12, 0.0)
    elev_penalty = -0.3 * (elevation / elevation.max())

    flood_p = (
        0.02
        + 0.50 * (rainfall / rainfall.max())
        + 0.40 * (water_level / water_level.max())
        + 0.25 * (river_discharge / river_discharge.max())
        + 0.10 * (humidity / 100.0)
        + 0.10 * historical_floods
        + soil_penalty + cover_penalty + infra_benefit + elev_penalty
    )
    flood_p = np.clip(flood_p, 0.0, 0.98)
    flood_occurred = rng.binomial(1, flood_p, n_samples)

    df = pd.DataFrame({
        "Latitude": lat.round(4),
        "Longitude": lon.round(4),
        "Rainfall_mm": rainfall.round(2),
        "Temperature_C": temperature.round(2),
        "Humidity_pct": humidity.round(2),
        "River_Discharge_m3_s": river_discharge.round(2),
        "Water_Level_m": water_level.round(3),
        "Elevation_m": elevation.round(1),
        "Land_Cover": land_cover,
        "Soil_Type": soil_type,
        "Population_Density": pop_density.round(1),
        "Infrastructure": infrastructure.astype(int),
        "Historical_Floods": historical_floods.astype(int),
        "Flood_Occurred": flood_occurred.astype(int),
    })

    out_path = Config.SYNTHETIC_DATA_DIR / Config.SYNTHETIC_DATASET_FILENAME
    Config.ensure_dirs()
    df.to_csv(out_path, index=False)
    logger.info("Generated synthetic dataset at %s (%d rows)", out_path, len(df))
    return df


def load_dataset(force_synthetic: bool = False) -> Tuple[pd.DataFrame, bool]:
    """
    Load the dataset for training.

    Returns
    -------
    (df, is_synthetic)
    """
    Config.ensure_dirs()
    real_path = _try_find_real_dataset()
    if real_path is not None and not force_synthetic:
        logger.info("Using real dataset: %s", real_path)
        df = pd.read_csv(real_path)
        return df, False

    synth_path = Config.SYNTHETIC_DATA_DIR / Config.SYNTHETIC_DATASET_FILENAME
    if synth_path.exists():
        logger.info("Using synthetic dataset (cached): %s", synth_path)
        return pd.read_csv(synth_path), True

    logger.warning(
        "Real dataset not found in %s; generating SYNTHETIC dataset for demo.",
        Config.RAW_DATA_DIR,
    )
    return generate_synthetic_dataset(), True


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
      - strip column names
      - coerce numerics
      - drop duplicates
      - reasonable clipping
      - impute missing medians/modes
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Try to map alternative column names to canonical ones
    rename_map = {}
    for col in df.columns:
        cl = col.lower().replace(" ", "_").replace("(", "").replace(")", "")
        canon = {
            "rainfall": "Rainfall_mm", "rainfall_mm": "Rainfall_mm",
            "temperature": "Temperature_C", "temperature_c": "Temperature_C",
            "humidity": "Humidity_pct", "humidity_pct": "Humidity_pct",
            "river_discharge": "River_Discharge_m3_s",
            "river_discharge_m3_s": "River_Discharge_m3_s",
            "water_level": "Water_Level_m", "water_level_m": "Water_Level_m",
            "elevation": "Elevation_m", "elevation_m": "Elevation_m",
            "land_cover": "Land_Cover",
            "soil_type": "Soil_Type",
            "population_density": "Population_Density",
            "infrastructure": "Infrastructure",
            "historical_floods": "Historical_Floods",
            "flood_occurred": "Flood_Occurred",
            "latitude": "Latitude", "longitude": "Longitude",
        }.get(cl)
        if canon and col != canon:
            rename_map[col] = canon
    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure canonical columns exist
    for c in EXPECTED_COLUMNS:
        if c not in df.columns:
            logger.warning("Column '%s' missing; adding as NaN", c)
            df[c] = np.nan

    df = df[EXPECTED_COLUMNS]

    # Type coercion
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in BINARY_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(0).round().clip(0, 1).astype(int)
    for c in CATEGORICAL_COLS:
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})

    # Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Dropped %d duplicate rows", before - len(df))

    # Clip physically implausible values
    df["Rainfall_mm"] = df["Rainfall_mm"].clip(lower=0, upper=2000)
    df["Temperature_C"] = df["Temperature_C"].clip(lower=-50, upper=60)
    df["Humidity_pct"] = df["Humidity_pct"].clip(lower=0, upper=100)
    df["River_Discharge_m3_s"] = df["River_Discharge_m3_s"].clip(lower=0, upper=100000)
    df["Water_Level_m"] = df["Water_Level_m"].clip(lower=0, upper=50)
    df["Elevation_m"] = df["Elevation_m"].clip(lower=-50, upper=9000)
    df["Population_Density"] = df["Population_Density"].clip(lower=0, upper=50000)
    df["Latitude"] = df["Latitude"].clip(lower=-90, upper=90)
    df["Longitude"] = df["Longitude"].clip(lower=-180, upper=180)

    # Impute missing
    for c in NUMERIC_COLS:
        if df[c].isna().any():
            med = df[c].median()
            df[c] = df[c].fillna(med)
    for c in CATEGORICAL_COLS:
        if df[c].isna().any():
            mode = df[c].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "Unknown"
            df[c] = df[c].fillna(fill)
    for c in BINARY_COLS:
        if df[c].isna().any():
            df[c] = df[c].fillna(0).astype(int)

    # Harmonize categorical levels
    df["Land_Cover"] = df["Land_Cover"].replace({
        "urban": "Urban", "forest": "Forest", "agricultural": "Agricultural",
        "water": "Water Body", "water body": "Water Body", "desert": "Desert",
    })
    df["Soil_Type"] = df["Soil_Type"].replace({
        "sandy": "Sandy", "clay": "Clay", "loam": "Loam", "silt": "Silt", "peat": "Peat",
    })
    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return a summary dict for the dashboard."""
    summary = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicates": 0,
        "numeric_means": {c: float(df[c].mean()) for c in NUMERIC_COLS if c in df.columns},
        "class_counts": {},
    }
    if "Risk_Level" in df.columns:
        summary["class_counts"] = df["Risk_Level"].value_counts().to_dict()
    elif "Flood_Occurred" in df.columns:
        summary["class_counts"] = df["Flood_Occurred"].value_counts().to_dict()
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df, is_synth = load_dataset()
    df = clean_data(df)
    Config.ensure_dirs()
    out = Config.PROCESSED_DATA_DIR / "cleaned.csv"
    df.to_csv(out, index=False)
    logger.info("Cleaned data saved to %s (synthetic=%s)", out, is_synth)
    print(dataset_summary(df))
