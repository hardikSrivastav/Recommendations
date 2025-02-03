from flask import Flask
from flask_cors import CORS
from application.routes.auth_routes import auth_bp
from application.routes.feedback_routes import feedback_bp
from application.routes.recommendation_routes import recommendation_bp
from application.routes.user_routes import user_bp
from application.routes.music_routes import music_bp
from application.database.database_service import DatabaseService

def create_app(testing=False):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    CORS(app)

    # Configure app
    app.config.from_object('config.Config')
    if testing:
        app.config.from_object('config.TestConfig')
    app.config['PROPAGATE_EXCEPTIONS'] = True
    app.config['JSON_SORT_KEYS'] = False  # Preserve JSON order

    # Initialize database
    app.db = DatabaseService(testing=testing)

    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(feedback_bp, url_prefix='/feedback')
    app.register_blueprint(recommendation_bp, url_prefix='/recommendations')
    app.register_blueprint(user_bp, url_prefix='/users')
    app.register_blueprint(music_bp, url_prefix='/music')

    return app 