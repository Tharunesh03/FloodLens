"""
Prediction service for FloodLens.

Loads the trained pipeline once and exposes a `predict` function that
takes a feature dictionary and returns:
  - risk_level string
  - numeric risk class
  - class probabilities
  - confidence (max probability)
  - contributing features (lightweight explanation)

Explanation strategy
--------------------
We use a simple, transparent contribution method:
  1. For numeric features, compute how far the supplied value is from the
     training-set mean in units of standard deviation, multiplied by the
     feature's global importance weight. Sign indicates whether that pushes
     risk up or down.
  2. For categorical / binary features, use the mean risk score (mean
     predicted probability of High+Critical) for that category as a
     contribution relative to the global mean.

This avoids SHAP's heavy deployment dependency while still producing
honest, model-aware explanations. All explanations are labelled
"contributing to the model's prediction", NOT as flood causes.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

from config import Config, RISK_LEVELS  # noqa: E402
from src.feature_engineering import RISK_LABELS  # noqa: E402

logger = logging.getLogger("floodlens.predict")

# Lazy-loaded singletons
_model: Optional[Any] = None
_metadata: Optional[Dict[str, Any]] = None
_train_stats: Optional[Dict[str, Any]] = None


def _load_artifacts():
    global _model, _metadata, _train_stats
    if _model is not None and _metadata is not None:
        return
    Config.ensure_dirs()
    model_path = Config.MODELS_DIR / Config.MODEL_FILE
    meta_path = Config.MODELS_DIR / Config.METADATA_FILE
    train_path = Config.PROCESSED_DATA_DIR / "engineered.csv"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python scripts/train_pipeline.py` first."
        )

    _model = joblib.load(model_path)
    with open(meta_path) as f:
        _metadata = json.load(f)

    # Load training data for statistics used in explanations
    if train_path.exists():
        df_train = pd.read_csv(train_path)
        proba = _model.predict_proba(df_train[_metadata["dataset"]["feature_names"]])
        high_crit = proba[:, 2] + proba[:, 3]
        df_train = df_train.assign(_p_high=high_crit)
        num_stats = {}
        for c in _metadata["dataset"]["numeric_features"]:
            num_stats[c] = {
                "mean": float(df_train[c].mean()),
                "std": float(df_train[c].std()) if float(df_train[c].std()) > 1e-9 else 1.0,
            }
        cat_stats = {}
        for c in _metadata["dataset"]["categorical_features"] + _metadata["dataset"]["binary_features"]:
            cat_stats[c] = {
                "global_mean": float(df_train["_p_high"].mean()),
                "level_means": df_train.groupby(c)["_p_high"].mean().to_dict(),
            }
        global_importance = None
        if _metadata.get("feature_importances"):
            global_importance = dict(zip(
                _metadata.get("feature_names_transformed") or [],
                _metadata["feature_importances"],
            ))
        _train_stats = {
            "num": num_stats,
            "cat": cat_stats,
            "global_p_high_mean": float(df_train["_p_high"].mean()),
            "feature_names": list(_metadata["dataset"]["feature_names"]),
            "global_importance": global_importance,
        }
    else:
        _train_stats = {
            "num": {}, "cat": {}, "global_p_high_mean": 0.5,
            "feature_names": list(_metadata["dataset"]["feature_names"]),
            "global_importance": None,
        }


# ---- Input validation ----

VALID_LAND_COVER = {"Urban", "Forest", "Agricultural", "Water Body", "Desert"}
VALID_SOIL_TYPE = {"Sandy", "Clay", "Loam", "Silt", "Peat"}

REQUIRED_INPUTS = [
    "Rainfall_mm", "Temperature_C", "Humidity_pct", "River_Discharge_m3_s",
    "Water_Level_m", "Elevation_m", "Land_Cover", "Soil_Type",
]
OPTIONAL_INPUTS = [
    "Population_Density", "Infrastructure", "Historical_Floods",
    "Latitude", "Longitude",
]

BOUNDS = {
    "Rainfall_mm": (0, 1000),
    "Temperature_C": (-20, 55),
    "Humidity_pct": (0, 100),
    "River_Discharge_m3_s": (0, 50000),
    "Water_Level_m": (0, 30),
    "Elevation_m": (-50, 9000),
    "Population_Density": (0, 50000),
    "Infrastructure": (0, 1),
    "Historical_Floods": (0, 1),
    "Latitude": (-90, 90),
    "Longitude": (-180, 180),
}


def validate_inputs(data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Validate and coerce user input. Returns (cleaned, errors)."""
    errors: List[str] = []
    cleaned: Dict[str, Any] = {}
    for key in REQUIRED_INPUTS:
        if key not in data or data[key] in ("", None):
            errors.append(f"Missing required field: {key}")
    if errors:
        return cleaned, errors

    for key, (lo, hi) in BOUNDS.items():
        if key not in data or data[key] in ("", None):
            continue
        try:
            val = float(data[key])
        except (TypeError, ValueError):
            errors.append(f"Field {key} must be numeric.")
            continue
        if val < lo or val > hi:
            errors.append(f"Field {key} must be between {lo} and {hi}.")
            continue
        cleaned[key] = val

    # categorical
    if "Land_Cover" in data and str(data["Land_Cover"]) not in VALID_LAND_COVER:
        errors.append(f"Land_Cover must be one of {sorted(VALID_LAND_COVER)}.")
    else:
        cleaned["Land_Cover"] = str(data.get("Land_Cover"))

    if "Soil_Type" in data and str(data["Soil_Type"]) not in VALID_SOIL_TYPE:
        errors.append(f"Soil_Type must be one of {sorted(VALID_SOIL_TYPE)}.")
    else:
        cleaned["Soil_Type"] = str(data.get("Soil_Type"))

    # optional numerics with sensible defaults
    cleaned.setdefault("Population_Density", 500.0)
    cleaned.setdefault("Infrastructure", 0)
    cleaned.setdefault("Historical_Floods", 0)
    cleaned.setdefault("Latitude", 22.0)
    cleaned.setdefault("Longitude", 78.0)

    return cleaned, errors


# ---- Feature construction (single-sample) ----

def build_model_features(cleaned: Dict[str, Any]) -> pd.DataFrame:
    """Compute engineered features from cleaned user inputs."""
    row = dict(cleaned)
    rainfall = row["Rainfall_mm"]
    water_level = row["Water_Level_m"]
    discharge = row["River_Discharge_m3_s"]
    humidity = row["Humidity_pct"]
    elevation = row["Elevation_m"]

    row["Rainfall_72h_proxy"] = float(np.sqrt(max(rainfall, 0)) * 4.5)
    row["Rainfall_x_Level"] = float(rainfall * water_level)
    row["Discharge_x_Humidity"] = float(discharge * humidity / 100.0)
    row["Elevation_inv"] = float(1.0 / (elevation + 10.0))

    _load_artifacts()
    assert _metadata is not None
    feats = _metadata["dataset"]["feature_names"]
    df = pd.DataFrame([{f: row.get(f, np.nan) for f in feats}])
    return df


# ---- Explanation ----

def _human_factor(direction: str, magnitude: str, feature_label: str) -> str:
    return f"{magnitude} {direction} {feature_label}".strip()


FEATURE_LABELS = {
    "Rainfall_mm": "rainfall",
    "River_Discharge_m3_s": "river discharge",
    "Water_Level_m": "water level",
    "Humidity_pct": "humidity",
    "Elevation_m": "elevation",
    "Temperature_C": "temperature",
    "Population_Density": "population density",
    "Infrastructure": "flood-control infrastructure",
    "Historical_Floods": "history of flooding at this location",
    "Rainfall_x_Level": "combined rainfall & water level",
    "Discharge_x_Humidity": "combined river flow & humidity",
    "Elevation_inv": "low elevation",
    "Rainfall_72h_proxy": "cumulative rainfall proxy",
    "Land_Cover": "land cover type",
    "Soil_Type": "soil type",
}


def explain(row: pd.DataFrame, proba: np.ndarray, top_k: int = 4) -> List[Dict[str, Any]]:
    """Return a list of contributing-factor dicts."""
    _load_artifacts()
    assert _train_stats is not None
    p_high = float(proba[0, 2] + proba[0, 3])
    global_mean = _train_stats["global_p_high_mean"]
    contributions = []

    for feat, stats in _train_stats["num"].items():
        if feat not in row.columns:
            continue
        val = float(row.iloc[0][feat])
        z = (val - stats["mean"]) / stats["std"]
        weight = 1.0
        # For elevation: higher elevation reduces risk
        if feat == "Elevation_m":
            z = -z
        contributions.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "z": z,
            "magnitude": abs(z),
            "direction": "elevates risk from" if z > 0 else "reduces risk via",
        })

    for feat, stats in _train_stats["cat"].items():
        if feat not in row.columns:
            continue
        val = row.iloc[0][feat]
        lvl_mean = float(stats["level_means"].get(val, stats["global_mean"]))
        delta = lvl_mean - stats["global_mean"]
        contributions.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "z": delta * 6.0,  # roughly rescale to ~z units
            "magnitude": abs(delta) * 6.0,
            "direction": "elevates risk from" if delta > 0 else "reduces risk via",
            "value": val,
        })

    # Sort by magnitude, take top_k
    contributions.sort(key=lambda c: c["magnitude"], reverse=True)
    top = contributions[:top_k]

    def mag_label(m: float) -> str:
        if m < 0.3:
            return "Minor"
        if m < 0.8:
            return "Notable"
        if m < 1.5:
            return "Significant"
        return "Major"

    def dir_label(d: str, feat: str) -> str:
        # More natural phrasing for elevation & infrastructure
        if feat == "Elevation_m" and "reduces" in d:
            return "risk reduced by higher"
        if feat == "Infrastructure":
            return "risk mitigated by presence of" if "reduces" in d else "risk without"
        return d

    factors = []
    for c in top:
        factors.append({
            "feature": c["feature"],
            "label": c["label"],
            "contribution": "up" if "elevates" in c["direction"] else "down",
            "magnitude": mag_label(c["magnitude"]),
            "text": f"{mag_label(c['magnitude'])} factor {dir_label(c['direction'], c['feature'])}: {c['label']}",
        })
    return factors


# ---- Main prediction ----

def predict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Run a prediction from a dict of raw inputs."""
    _load_artifacts()
    assert _model is not None and _metadata is not None
    cleaned, errors = validate_inputs(data)
    if errors:
        return {"ok": False, "errors": errors}

    row = build_model_features(cleaned)
    proba = _model.predict_proba(row)
    cls_idx = int(np.argmax(proba, axis=1)[0])
    label = RISK_LEVELS[cls_idx]
    confidence = float(np.max(proba))
    proba_dict = {RISK_LEVELS[i]: float(proba[0, i]) for i in range(len(RISK_LABELS))}
    factors = explain(row, proba)

    precaution = precautions_for(label)
    interpretation = interpretation_for(label)

    return {
        "ok": True,
        "model_version": _metadata["model_version"],
        "is_synthetic_data": _metadata["is_synthetic_data"],
        "risk_level": label,
        "risk_index": cls_idx,
        "confidence": confidence,
        "probabilities": proba_dict,
        "important_factors": factors,
        "input_summary": {
            "Rainfall (mm)": cleaned["Rainfall_mm"],
            "Temperature (°C)": cleaned["Temperature_C"],
            "Humidity (%)": cleaned["Humidity_pct"],
            "River Discharge (m³/s)": cleaned["River_Discharge_m3_s"],
            "Water Level (m)": cleaned["Water_Level_m"],
            "Elevation (m)": cleaned["Elevation_m"],
            "Land Cover": cleaned["Land_Cover"],
            "Soil Type": cleaned["Soil_Type"],
            "Flood-control Infrastructure": "Yes" if cleaned.get("Infrastructure") else "No",
            "Historical Floods": "Yes" if cleaned.get("Historical_Floods") else "No",
        },
        "interpretation": interpretation,
        "precautions": precaution,
    }


def interpretation_for(level: str) -> str:
    return {
        "Low": "Flood risk is currently estimated to be low based on the supplied conditions. Routine awareness is sufficient.",
        "Moderate": "Conditions suggest a moderate flood risk. Stay alert and monitor local weather updates.",
        "High": "Flood risk is estimated to be high. Avoid low-lying areas and monitor official weather / emergency information.",
        "Critical": "Conditions indicate elevated modelled risk. Follow official local emergency guidance and prepare to act.",
    }[level]


def precautions_for(level: str) -> List[str]:
    base = [
        "Monitor official meteorological / disaster-management channels.",
        "Do not use this tool as an official emergency warning.",
    ]
    if level == "Low":
        return ["No immediate action required."] + base
    if level == "Moderate":
        return [
            "Prepare an emergency kit with essentials.",
            "Clear nearby drains of debris.",
        ] + base
    if level == "High":
        return [
            "Avoid driving or walking through flooded areas.",
            "Move valuables to higher ground if safe to do so.",
            "Charge devices and keep emergency contacts accessible.",
        ] + base
    return [
        "Follow evacuation instructions if issued by authorities.",
        "Move to higher ground immediately if instructed or in danger.",
        "Avoid rivers, drains, and low-lying zones.",
        "Keep communication devices charged and stay informed.",
    ] + base


def get_metadata() -> Dict[str, Any]:
    _load_artifacts()
    assert _metadata is not None
    # Attach dataset status at runtime
    try:
        from config import real_dataset_status
        _metadata["_dataset_status"] = real_dataset_status()
    except Exception:
        _metadata["_dataset_status"] = {"available": False}
    return _metadata
