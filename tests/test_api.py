"""Tests for Flask API endpoints and basic page responses."""
from __future__ import annotations

import json


def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert body["service"] == "FloodLens"


def test_predict_endpoint_valid(client):
    payload = {
        "Rainfall_mm": 90, "Temperature_C": 26, "Humidity_pct": 80,
        "River_Discharge_m3_s": 2000, "Water_Level_m": 4, "Elevation_m": 120,
        "Land_Cover": "Agricultural", "Soil_Type": "Clay",
        "Infrastructure": 1, "Historical_Floods": 0,
    }
    r = client.post("/api/predict", data=json.dumps(payload),
                    content_type="application/json")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["risk_level"] in ("Low", "Moderate", "High", "Critical")
    assert "probabilities" in body
    assert "important_factors" in body


def test_predict_endpoint_invalid_input(client):
    payload = {"Rainfall_mm": "not-a-number"}
    r = client.post("/api/predict", data=json.dumps(payload),
                    content_type="application/json")
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False
    assert "errors" in body


def test_predict_endpoint_non_json(client):
    r = client.post("/api/predict", data="oops", content_type="text/plain")
    assert r.status_code == 400


def test_home_page_loads(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"FloodLens" in r.data


def test_predict_page_loads(client):
    r = client.get("/predict")
    assert r.status_code == 200


def test_dashboard_loads(client):
    r = client.get("/dashboard")
    assert r.status_code == 200


def test_about_page_loads(client):
    r = client.get("/about")
    assert r.status_code == 200


def test_metadata_endpoint(client):
    r = client.get("/api/metadata")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert "selected_model" in body["metadata"]
