"""
Feature engineering and target construction for FloodLens.

Target definition
-----------------
The real dataset ships only with a binary `Flood_Occurred` flag. We derive a
4-class ordinal flood-risk target (Low / Moderate / High / Critical) by
constructing a continuous Risk Score from physically motivated drivers, then
binning it while ensuring every flood event is labelled at least High/Critical.

Risk Score (0..1) is a weighted combination of:
  * Rainfall percentile        (primary driver)
  * River discharge percentile
  * Water-level percentile
  * Humidity percentile
  * Low elevation              (higher risk at low elevation)
  * Historical flood flag      (memory of past events)
  * Soil drainage penalty      (clay/silt/peat retain water)
  * Land-cover penalty         (water bodies, urban impermeable surfaces)
  * Infrastructure benefit     (mitigation)

The target is built as follows:
  score < 0.30                → Low
  0.30 ≤ score < 0.50         → Moderate
  0.50 ≤ score < 0.75         → High
  score ≥ 0.75 OR Flood_Occurred=1 with high drivers → Critical

All rows with Flood_Occurred=1 are promoted to at least High.

This is a composite hazard indicator, not a ground-truth severity label; see
docs/PROJECT_REPORT.md for discussion and limitations.
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import Config  # noqa: E402

logger = logging.getLogger("floodlens.features")

# Features actually fed to the ML model
NUMERIC_FEATURES = [
    "Rainfall_mm",
    "Temperature_C",
    "Humidity_pct",
    "River_Discharge_m3_s",
    "Water_Level_m",
    "Elevation_m",
    "Population_Density",
    "Rainfall_x_Level",        # engineered interaction
    "Discharge_x_Humidity",    # engineered interaction
    "Elevation_inv",           # 1/(elev+10)
    "Rainfall_72h_proxy",      # proxy cumulative rainfall
]
CATEGORICAL_FEATURES = ["Land_Cover", "Soil_Type"]
BINARY_FEATURES = ["Infrastructure", "Historical_Floods"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

TARGET_COL = "Risk_Level"
RISK_LABELS = ["Low", "Moderate", "High", "Critical"]


SOIL_DRAINAGE_PENALTY = {
    "Sandy": 0.00, "Loam": 0.05, "Silt": 0.12, "Clay": 0.18, "Peat": 0.15,
}
LAND_COVER_PENALTY = {
    "Forest": -0.05, "Agricultural": 0.00, "Desert": -0.05,
    "Urban": 0.10, "Water Body": 0.20,
}


def _percentile_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Works on a cleaned DataFrame."""
    df = df.copy()

    # Proxy for cumulative/antecedent rainfall: clip then sqrt-transform to capture
    # non-linear response; we don't have time series so we approximate.
    df["Rainfall_72h_proxy"] = np.sqrt(df["Rainfall_mm"].clip(lower=0)) * 4.5

    # Interactions
    df["Rainfall_x_Level"] = df["Rainfall_mm"] * df["Water_Level_m"]
    df["Discharge_x_Humidity"] = df["River_Discharge_m3_s"] * df["Humidity_pct"] / 100.0
    df["Elevation_inv"] = 1.0 / (df["Elevation_mm"] if "Elevation_mm" in df.columns else df["Elevation_m"] + 10.0)

    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Construct the 4-class Risk_Level target from drivers + Flood_Occurred."""
    df = df.copy()

    p_rain = _percentile_rank(df["Rainfall_mm"])
    p_dis = _percentile_rank(df["River_Discharge_m3_s"])
    p_lvl = _percentile_rank(df["Water_Level_m"])
    p_hum = _percentile_rank(df["Humidity_pct"])
    p_elev_inv = _percentile_rank(1.0 / (df["Elevation_m"] + 10.0))

    soil_p = df["Soil_Type"].map(SOIL_DRAINAGE_PENALTY).fillna(0.0)
    cover_p = df["Land_Cover"].map(LAND_COVER_PENALTY).fillna(0.0)
    infra_b = np.where(df["Infrastructure"] == 1, -0.10, 0.0)
    hist_b = np.where(df["Historical_Floods"] == 1, 0.10, 0.0)

    score = (
        0.35 * p_rain
        + 0.20 * p_lvl
        + 0.15 * p_dis
        + 0.10 * p_hum
        + 0.10 * p_elev_inv
        + soil_p + cover_p + infra_b + hist_b
    )
    score = score.clip(0.0, 1.0)
    df["Risk_Score"] = score

    # Bin into 4 classes
    df["Risk_Level"] = pd.cut(
        score,
        bins=[-0.001, 0.30, 0.50, 0.75, 1.001],
        labels=RISK_LABELS,
    ).astype(str)

    # Ensure flood events are at least High; critical if score >= 0.6
    flood_mask = df["Flood_Occurred"] == 1
    df.loc[flood_mask & (df["Risk_Score"] >= 0.6), "Risk_Level"] = "Critical"
    df.loc[flood_mask & (df["Risk_Score"] < 0.6), "Risk_Level"] = "High"

    # Map string labels to ints for sklearn
    df["Risk_Level_Int"] = df["Risk_Level"].map({lbl: i for i, lbl in enumerate(RISK_LABELS)}).astype(int)
    return df


def get_feature_names() -> List[str]:
    return list(MODEL_FEATURES)


def engineer_pipeline(df_clean: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Build features + target. Returns (engineered_df, feature_names)."""
    df = build_features(df_clean)
    df = build_target(df)
    return df, get_feature_names()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.data_processing import load_dataset, clean_data  # noqa: E402
    df, is_synth = load_dataset()
    df = clean_data(df)
    df, feats = engineer_pipeline(df)
    Config.ensure_dirs()
    out = Config.PROCESSED_DATA_DIR / "engineered.csv"
    df.to_csv(out, index=False)
    logger.info("Engineered data saved to %s", out)
    logger.info("Feature list: %s", feats)
    logger.info("Target distribution:\n%s", df["Risk_Level"].value_counts())
