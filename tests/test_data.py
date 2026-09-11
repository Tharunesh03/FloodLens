"""Tests for data processing and feature engineering."""
from __future__ import annotations

import pandas as pd

from src.data_processing import clean_data, generate_synthetic_dataset
from src.feature_engineering import engineer_pipeline, RISK_LABELS


def test_synthetic_dataset_generation():
    df = generate_synthetic_dataset(n_samples=500, seed=0)
    assert len(df) == 500
    for col in [
        "Rainfall_mm", "Temperature_C", "Humidity_pct", "River_Discharge_m3_s",
        "Water_Level_m", "Elevation_m", "Land_Cover", "Soil_Type",
        "Population_Density", "Infrastructure", "Historical_Floods", "Flood_Occurred",
    ]:
        assert col in df.columns


def test_clean_data_handles_missing_and_outliers():
    df = generate_synthetic_dataset(n_samples=200, seed=1)
    # Inject an outlier and NaN
    df.loc[0, "Rainfall_mm"] = 10**9
    df.loc[1, "Humidity_pct"] = None
    cleaned = clean_data(df)
    assert cleaned["Rainfall_mm"].max() <= 2000
    assert cleaned["Humidity_pct"].isna().sum() == 0
    assert len(cleaned) == len(df)  # no rows lost (imputation, not deletion)


def test_feature_engineering_produces_target_and_classes():
    df = generate_synthetic_dataset(n_samples=500, seed=2)
    cleaned = clean_data(df)
    eng, feats = engineer_pipeline(cleaned)
    for f in feats:
        assert f in eng.columns, f"missing feature {f}"
    assert "Risk_Level" in eng.columns
    assert "Risk_Level_Int" in eng.columns
    assert set(eng["Risk_Level"].unique()).issubset(set(RISK_LABELS))
    # Every flood row should be at least High
    flood_levels = eng.loc[eng["Flood_Occurred"] == 1, "Risk_Level"].unique()
    for lbl in flood_levels:
        assert lbl in ("High", "Critical")
