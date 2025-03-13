from flask import Flask

def create_app():
    app = Flask(__name__)

    # Configure folders for file uploads and results
    app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
    app.config['RESULT_FOLDER'] = 'app/static/results'

    # Import and register the blueprint
    from .routes import bp
    app.register_blueprint(bp)

    return app
