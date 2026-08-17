"""
Main Flask application entrypoint for AI Therapy Notes Maker.

Configures Flask application instance, registers API blueprints,
and defines client-facing routes.
"""

from flask import Flask, render_template
from routes import api_bp
from config.config import UPLOAD_FOLDER, MAX_CONTENT_LENGTH, logger



def create_app() -> Flask:
    """
    Application factory for AI Therapy Notes Maker.

    Returns:
        Flask: Configured Flask application instance.
    """
    app = Flask(__name__)
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 3600
    app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    # Register API Blueprint
    app.register_blueprint(api_bp)

    @app.route("/")
    def home():
        """Render the main interactive client interface."""
        return render_template("index.html")

    @app.route("/favicon.ico")
    def favicon():
        """Handle browser favicon request cleanly to avoid 404 logs."""
        return "", 204

    logger.info("AI Therapy Notes Maker application initialized successfully.")
    return app


# WSGI application instance for serverless/production runtimes (e.g. Vercel, Gunicorn)
app = create_app()

if __name__ == "__main__":
    # use_reloader=False prevents watchdog socket issues on Windows
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)