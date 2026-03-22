"""
app.py  –  Flask Application Factory & Routes
==============================================
Registers all HTTP routes used by the frontend.

Routes
------
GET  /                      → Serve the single-page HTML interface
POST /api/process-frame     → Accept a base-64 webcam frame, run MediaPipe,
                              return landmark JSON for the chosen face side
"""

from flask import Flask, render_template, send_from_directory
from flask_cors import CORS
import os


def create_app() -> Flask:
    """
    Application factory.

    Returns a fully configured Flask instance with:
      - CORS enabled for local development
      - All routes registered
    """
    # Point Flask at the static/ and templates/ folders
    base_dir = os.path.dirname(__file__)

    app = Flask(
        __name__,
        template_folder=os.path.join(base_dir, "templates"),
        static_folder=os.path.join(base_dir, "static"),
    )

    CORS(app)  # Allow the browser to call /api/* without CORS errors

    # ── Register blueprints ──────────────────────────────────────────────
    from routes.landmarks import landmarks_bp
    app.register_blueprint(landmarks_bp)

    from routes.eeg_predict import eeg_predict_bp
    app.register_blueprint(eeg_predict_bp)

    # ── Root route ───────────────────────────────────────────────────────
    @app.route("/")
    def index():
        """Serve the main single-page application."""
        return render_template("index.html")

    return app
