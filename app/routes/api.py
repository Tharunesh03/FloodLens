"""JSON API endpoints for FloodLens."""
from __future__ import annotations

import logging
from flask import Blueprint, jsonify, request

from src.predict import predict as run_prediction, get_metadata
from app.services.database import record_prediction

api_bp = Blueprint("api", __name__)
logger = logging.getLogger("floodlens.api")


@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "FloodLens",
        "version": get_metadata().get("model_version", "unknown") if _model_ready() else "unknown",
    })


@api_bp.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"ok": False, "error": "Request must be JSON"}), 400
    payload = request.get_json(silent=True) or {}
    result = run_prediction(payload)
    if not result.get("ok"):
        return jsonify(result), 400
    try:
        record_prediction(result, source="api")
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to record prediction history: %s", exc)
    return jsonify(result), 200


@api_bp.route("/metadata", methods=["GET"])
def metadata():
    if not _model_ready():
        return jsonify({"ok": False, "error": "Model not trained."}), 503
    meta = get_metadata()
    safe = {
        "model_version": meta.get("model_version"),
        "trained_at": meta.get("trained_at"),
        "selected_model": meta["model_selection"]["selected_model"],
        "is_synthetic_data": meta.get("is_synthetic_data"),
        "test_metrics": meta.get("test_metrics"),
        "dataset": {
            "rows": meta["dataset"]["rows"],
            "class_distribution": meta["dataset"]["class_distribution"],
            "feature_names": meta["dataset"]["feature_names"],
            "risk_labels": meta["dataset"]["risk_labels"],
        },
    }
    return jsonify({"ok": True, "metadata": safe})


def _model_ready() -> bool:
    try:
        get_metadata()
        return True
    except Exception:
        return False
