from flask import Flask, render_template
from api import api_bp
from config import UPLOAD_FOLDER, logger

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
    app.run(debug=True)