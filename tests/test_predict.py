"""Tests for prediction service."""
from __future__ import annotations

import importlib

import pytest

from src import predict as predict_mod
from src.predict import predict, validate_inputs


def _reload_predict():
    # Reset lazy singletons so tests pick up test-configured Config paths
    predict_mod._model = None
    predict_mod._metadata = None
    predict_mod._train_stats = None


def test_validate_inputs_rejects_missing_fields():
    cleaned, errors = validate_inputs({})
    assert errors
    assert any("Missing" in e for e in errors)


def test_validate_inputs_rejects_out_of_range():
    data = {
        "Rainfall_mm": 999999, "Temperature_C": 25, "Humidity_pct": 70,
        "River_Discharge_m3_s": 100, "Water_Level_m": 2, "Elevation_m": 100,
        "Land_Cover": "Urban", "Soil_Type": "Loam",
    }
    _, errors = validate_inputs(data)
    assert errors


def test_predict_end_to_end(app_with_model):
    _reload_predict()
    data = {
        "Rainfall_mm": 120, "Temperature_C": 28, "Humidity_pct": 88,
        "River_Discharge_m3_s": 3500, "Water_Level_m": 6.5, "Elevation_m": 80,
        "Land_Cover": "Agricultural", "Soil_Type": "Clay",
        "Infrastructure": 0, "Historical_Floods": 1,
    }
    res = predict(data)
    assert res["ok"] is True
    assert res["risk_level"] in ("Low", "Moderate", "High", "Critical")
    assert 0.0 <= res["confidence"] <= 1.0
    assert set(res["probabilities"].keys()) == {"Low", "Moderate", "High", "Critical"}
    assert abs(sum(res["probabilities"].values()) - 1.0) < 1e-6
    assert isinstance(res["important_factors"], list)
    assert len(res["important_factors"]) >= 1


def test_predict_invalid_categorical(app_with_model):
    _reload_predict()
    data = {
        "Rainfall_mm": 50, "Temperature_C": 25, "Humidity_pct": 60,
        "River_Discharge_m3_s": 500, "Water_Level_m": 2, "Elevation_m": 200,
        "Land_Cover": "Mars", "Soil_Type": "Loam",
    }
    res = predict(data)
    assert res["ok"] is False
    assert any("Land_Cover" in e for e in res["errors"])
