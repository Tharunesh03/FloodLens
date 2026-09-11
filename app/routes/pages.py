"""Page routes for FloodLens: home, predict, dashboard, history, about, dataset, etc."""
from __future__ import annotations

import logging
from threading import Lock

from flask import (
    Blueprint, render_template, request, redirect, url_for,
    flash, current_app, jsonify,
)
from flask_wtf import FlaskForm
from wtforms import FloatField, SelectField, SubmitField
from wtforms.validators import InputRequired, NumberRange, Optional as OptionalVal

from config import Config, RISK_LEVELS
from src.predict import (
    predict as run_prediction, get_metadata,
)
from app.services.database import (
    record_prediction, recent_predictions, count_predictions,
    risk_counts, clear_history,
)
from src.visualize import generate_dashboard_charts

pages_bp = Blueprint("pages", __name__)
logger = logging.getLogger("floodlens.pages")

_chart_lock = Lock()


def _ensure_charts():
    with _chart_lock:
        if getattr(current_app, "charts_cache", None) is None:
            try:
                current_app.charts_cache = generate_dashboard_charts()
            except Exception as exc:
                logger.exception("Chart generation failed: %s", exc)
                current_app.charts_cache = {"available": False, "error": str(exc)}
    return current_app.charts_cache


def _model_ready() -> bool:
    try:
        get_metadata()
        return True
    except Exception:
        return False


# ---- Forms ----

class PredictForm(FlaskForm):
    Rainfall_mm = FloatField("Rainfall (mm)", validators=[InputRequired(), NumberRange(min=0, max=1000)], default=80.0)
    Temperature_C = FloatField("Temperature (°C)", validators=[InputRequired(), NumberRange(min=-20, max=55)], default=27.0)
    Humidity_pct = FloatField("Humidity (%)", validators=[InputRequired(), NumberRange(min=0, max=100)], default=75.0)
    River_Discharge_m3_s = FloatField("River Discharge (m³/s)", validators=[InputRequired(), NumberRange(min=0, max=50000)], default=1200.0)
    Water_Level_m = FloatField("Water Level (m)", validators=[InputRequired(), NumberRange(min=0, max=30)], default=3.5)
    Elevation_m = FloatField("Elevation (m)", validators=[InputRequired(), NumberRange(min=-50, max=9000)], default=250.0)
    Population_Density = FloatField("Population Density (per km²)", validators=[OptionalVal(), NumberRange(min=0, max=50000)], default=800.0)
    Land_Cover = SelectField("Land Cover", choices=[(c, c) for c in ["Agricultural", "Urban", "Forest", "Water Body", "Desert"]], default="Agricultural")
    Soil_Type = SelectField("Soil Type", choices=[(c, c) for c in ["Loam", "Clay", "Sandy", "Silt", "Peat"]], default="Loam")
    Infrastructure = SelectField("Flood-control Infrastructure", choices=[(0, "No"), (1, "Yes")], coerce=int, default=0)
    Historical_Floods = SelectField("Historical Floods at Location", choices=[(0, "No"), (1, "Yes")], coerce=int, default=0)
    submit = SubmitField("Estimate Flood Risk")


# ---- Routes ----

@pages_bp.route("/")
def home():
    return render_template("pages/home.html")


@pages_bp.route("/predict", methods=["GET", "POST"])
def predict_page():
    if not _model_ready():
        return render_template("pages/model_missing.html")
    form = PredictForm()
    if request.method == "GET":
        # Set sensible defaults for an empty form
        for f in [form.Rainfall_mm, form.Temperature_C, form.Humidity_pct,
                  form.River_Discharge_m3_s, form.Water_Level_m, form.Elevation_m,
                  form.Population_Density]:
            if f.data is None:
                f.data = f.default
    result = None
    if form.validate_on_submit():
        data = {
            "Rainfall_mm": form.Rainfall_mm.data,
            "Temperature_C": form.Temperature_C.data,
            "Humidity_pct": form.Humidity_pct.data,
            "River_Discharge_m3_s": form.River_Discharge_m3_s.data,
            "Water_Level_m": form.Water_Level_m.data,
            "Elevation_m": form.Elevation_m.data,
            "Population_Density": form.Population_Density.data,
            "Land_Cover": form.Land_Cover.data,
            "Soil_Type": form.Soil_Type.data,
            "Infrastructure": int(form.Infrastructure.data),
            "Historical_Floods": int(form.Historical_Floods.data),
        }
        try:
            result = run_prediction(data)
        except Exception as exc:
            logger.exception("Prediction failed")
            flash(f"Prediction error: {exc}", "danger")
            result = None
        if result and result.get("ok"):
            try:
                record_prediction(result, source="web")
            except Exception as exc:
                logger.warning("Failed to write history: %s", exc)
        elif result and not result.get("ok"):
            for err in result.get("errors", []):
                flash(err, "danger")
            result = None
    return render_template("pages/predict.html", form=form, result=result)


@pages_bp.route("/dashboard")
def dashboard():
    charts = _ensure_charts()
    meta = None
    try:
        meta = get_metadata()
    except Exception:
        meta = None
    history_stats = {
        "total_predictions": count_predictions(),
        "risk_counts": risk_counts(),
    }
    return render_template("pages/dashboard.html", charts=charts, meta=meta,
                           history_stats=history_stats)


@pages_bp.route("/history")
def history():
    records = recent_predictions(limit=200)
    return render_template("pages/history.html", records=records)


@pages_bp.route("/history/clear", methods=["POST"])
def history_clear():
    clear_history()
    flash("Prediction history cleared.", "info")
    return redirect(url_for("pages.history"))


@pages_bp.route("/about")
def about():
    meta = None
    try:
        meta = get_metadata()
    except Exception:
        meta = None
    return render_template("pages/about.html", meta=meta)


@pages_bp.route("/methodology")
def methodology():
    return render_template("pages/methodology.html")


@pages_bp.route("/api")
def api_docs():
    return render_template("pages/api_docs.html")


@pages_bp.route("/health")
def health_alias():
    from flask import jsonify
    try:
        version = get_metadata().get("model_version", "unknown")
    except Exception:
        version = "unknown"
    return jsonify({"status": "ok", "service": "FloodLens", "version": version})


@pages_bp.route("/dataset")
def dataset_page():
    """Admin/dataset management page."""
    from config import real_dataset_status
    from app.routes.api import _retrain_state
    ds = real_dataset_status()
    meta = None
    try:
        meta = get_metadata()
    except Exception:
        meta = None
    return render_template("pages/dataset.html", ds=ds, meta=meta, retrain=_retrain_state)
