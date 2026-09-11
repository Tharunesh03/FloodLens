"""
FloodLens Flask application factory.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Flask

from config import Config


def create_app(config_override: dict | None = None) -> Flask:
    Config.ensure_dirs()
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
        instance_path=str(Config.INSTANCE_DIR),
    )
    app.config.from_object(Config)
    app.config["SECRET_KEY"] = Config.SECRET_KEY
    app.config["WTF_CSRF_ENABLED"] = True
    if config_override:
        app.config.update(config_override)

    # Ensure instance folder exists
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Register blueprints
    from app.routes.pages import pages_bp
    from app.routes.api import api_bp
    app.register_blueprint(pages_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    # Initialize DB
    from app.services.database import init_db, close_db
    with app.app_context():
        init_db()
    app.teardown_appcontext(close_db)

    # Make chart generation lazy on first dashboard hit
    app.charts_cache = None

    @app.context_processor
    def inject_globals():
        from config import real_dataset_status
        try:
            dataset_status = real_dataset_status()
        except Exception:
            dataset_status = {"available": False}
        return {
            "APP_NAME": "FloodLens",
            "TAGLINE": "AI-Based Flood Risk Prediction & Early Warning",
            "RISK_COLORS": Config.RISK_COLORS,
            "RISK_LEVELS": Config.RISK_LEVELS,
            "GEO_COVERAGE": Config.GEO_COVERAGE,
            "DATASET_STATUS": dataset_status,
        }

    @app.errorhandler(404)
    def not_found(e):
        from flask import render_template
        return render_template("pages/error.html", code=404,
                               message="The page you requested was not found."), 404

    @app.errorhandler(500)
    def server_error(e):
        from flask import render_template
        return render_template("pages/error.html", code=500,
                               message="Internal server error. Please try again."), 500

    return app
