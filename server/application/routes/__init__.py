from flask import Flask
from flask_cors import CORS
from application.database.database_service import DatabaseService

def create_app(testing: bool = False):
    app = Flask(__name__)
    
    # Configure CORS
    cors = CORS(
        app,
        resources={r"/api/*": {
            "origins": ["http://localhost:3001"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-Session-ID"],
            "expose_headers": ["X-Session-ID"],  # Allow client to read this header
            "supports_credentials": True
        }}
    )

    # Import blueprints
    from .user_routes import user_bp
    from .music_routes import music_bp
    from .recommendation_routes import recommendation_bp
    from .feedback_routes import feedback_bp
    from .auth_routes import auth_bp

    # Initialize database service
    db_service = DatabaseService(testing=testing)

    # Initialize blueprints with database service
    for route in [user_bp, music_bp, recommendation_bp, feedback_bp, auth_bp]:
        if hasattr(route, 'init_db'):
            route.init_db(testing=testing)

    # Register blueprints
    app.register_blueprint(user_bp, url_prefix='/api/users')
    app.register_blueprint(music_bp, url_prefix='/api/music')
    app.register_blueprint(recommendation_bp, url_prefix='/api/recommendations')
    app.register_blueprint(feedback_bp, url_prefix='/api/feedback')
    app.register_blueprint(auth_bp, url_prefix='/api')

    return app 