from flask import Flask, render_template
from api import api_bp
from config.config import UPLOAD_FOLDER, logger

def create_app():
    app = Flask(__name__)
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3600
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    # Register Blueprints
    app.register_blueprint(api_bp)

    @app.route('/')
    def home():
        return render_template('index.html')

    logger.info("Application started.")
    return app

if __name__ == "__main__":
    app = create_app()
    # use_reloader=False prevents watchdog from killing in-flight requests on Windows
    # (WinError 10038 — socket closed during reload triggered by numba/torch file changes)
    app.run(debug=True, use_reloader=False)