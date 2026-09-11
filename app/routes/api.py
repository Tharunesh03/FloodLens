"""JSON API endpoints for FloodLens."""
from __future__ import annotations

import io
import logging
import threading
import traceback

from flask import Blueprint, jsonify, request

from src.predict import predict as run_prediction, get_metadata
from app.services.database import record_prediction

api_bp = Blueprint("api", __name__)
logger = logging.getLogger("floodlens.api")

_retrain_lock = threading.Lock()
_retrain_state = {"running": False, "last_log": None, "last_ok": None}


@api_bp.route("/health", methods=["GET"])
def health():
    try:
        version = get_metadata().get("model_version", "unknown")
    except Exception:
        version = "unknown"
    return jsonify({"status": "ok", "service": "FloodLens", "version": version})


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
    try:
        meta = get_metadata()
    except Exception:
        return jsonify({"ok": False, "error": "Model not trained."}), 503
    safe = {
        "model_version": meta.get("model_version"),
        "trained_at": meta.get("trained_at"),
        "selected_model": meta["model_selection"]["selected_model"],
        "is_synthetic_data": meta.get("is_synthetic_data"),
        "dataset_status": meta.get("_dataset_status", {"available": False}),
        "test_metrics": meta.get("test_metrics"),
        "dataset": {
            "rows": meta["dataset"]["rows"],
            "class_distribution": meta["dataset"]["class_distribution"],
            "feature_names": meta["dataset"]["feature_names"],
            "risk_labels": meta["dataset"]["risk_labels"],
        },
    }
    return jsonify({"ok": True, "metadata": safe})


@api_bp.route("/dataset_status", methods=["GET"])
def dataset_status():
    from config import real_dataset_status
    return jsonify({
        "ok": True,
        "dataset": real_dataset_status(),
        "retrain": _retrain_state,
    })


@api_bp.route("/retrain", methods=["POST"])
def retrain():
    """Trigger model retraining in a background thread."""
    if _retrain_lock.locked():
        return jsonify({"ok": False, "error": "Retraining already in progress."}), 409

    def _do_retrain(force_synthetic: bool):
        from src.train import run_training
        _retrain_state["running"] = True
        _retrain_state["last_log"] = "Starting training..."
        try:
            buf = io.StringIO()
            sh = logging.StreamHandler(buf)
            sh.setLevel(logging.INFO)
            logging.getLogger().addHandler(sh)
            try:
                meta = run_training(force_synthetic=force_synthetic)
            finally:
                logging.getLogger().removeHandler(sh)
            _retrain_state["last_log"] = (
                f"Done. {meta['model_selection']['selected_model']} | "
                f"f1_macro={meta['test_metrics']['f1_macro']:.3f} | "
                f"synthetic={meta['is_synthetic_data']}"
            )
            _retrain_state["last_ok"] = True
        except Exception as e:
            _retrain_state["last_log"] = "Error: " + str(e) + "\n" + traceback.format_exc()
            _retrain_state["last_ok"] = False
        finally:
            _retrain_state["running"] = False

    payload = request.get_json(silent=True) or {}
    force_synth = bool(payload.get("force_synthetic", False))
    t = threading.Thread(target=_do_retrain, args=(force_synth,), daemon=True)
    t.start()
    return jsonify({"ok": True, "status": "started"})
