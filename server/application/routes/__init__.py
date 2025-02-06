from flask import Flask, request
from flask_cors import CORS
from application.database.database_service import DatabaseService

def create_app(testing: bool = False):
    app = Flask(__name__)
    
    # Configure CORS
    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": ["http://localhost:3000", "http://localhost:3001"],
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": [
                    "Content-Type",
                    "Authorization",
                    "X-Session-ID",
                    "Accept",
                    "Origin",
                    "X-Requested-With"
                ],
                "expose_headers": ["X-Session-ID"],
                "supports_credentials": True,
                "send_wildcard": False,
                "max_age": 86400,  # Cache preflight requests for 24 hours
                "vary_header": True,
                "allow_credentials": True
            }
        }
    )

    # Add CORS headers to all responses
    @app.after_request
    def add_cors_headers(response):
        origin = request.headers.get('Origin')
        if origin in ["http://localhost:3000", "http://localhost:3001"]:
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Credentials'] = 'true'
            if request.method == 'OPTIONS':
                response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
                response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Session-ID, Accept, Origin, X-Requested-With'
                response.headers['Access-Control-Max-Age'] = '86400'
        return response

    # Import blueprints
    from .user_routes import user_bp
    from .music_routes import music_bp
    from .recommendation_routes import recommendation_bp
    from .feedback_routes import feedback_bp
    from .auth_routes import auth_bp
    from .training_routes import training_bp

    # Initialize database service
    db_service = DatabaseService(testing=testing)

    # Initialize blueprints with database service
    for route in [user_bp, music_bp, recommendation_bp, feedback_bp, auth_bp, training_bp]:
        if hasattr(route, 'init_db'):
            route.init_db(testing=testing)

    # Register blueprints
    app.register_blueprint(user_bp, url_prefix='/api/users')
    app.register_blueprint(music_bp, url_prefix='/api/music')
    app.register_blueprint(recommendation_bp, url_prefix='/api/recommendations')
    app.register_blueprint(feedback_bp, url_prefix='/api/feedback')
    app.register_blueprint(auth_bp, url_prefix='/api')
    app.register_blueprint(training_bp, url_prefix='/api/training')

    return app

# Import all route blueprints
from flask import request
from .auth_routes import auth_bp
from .feedback_routes import feedback_bp
from .recommendation_routes import recommendation_bp
from .user_routes import user_bp
from .music_routes import music_bp
from .training_routes import training_bp

# Initialize route modules that need initialization
def init_routes(testing=False):
    """Initialize all route modules that require initialization."""
    from .auth_routes import init_db as init_auth
    from .music_routes import init_services as init_music
    from .recommendation_routes import init_db as init_recommendations
    from .feedback_routes import init_db as init_feedback
    from .training_routes import init_services as init_training

    # Add CORS headers to all blueprint responses
    for bp in [auth_bp, feedback_bp, recommendation_bp, user_bp, music_bp, training_bp]:
        @bp.after_request
        def add_cors_headers(response):
            origin = request.headers.get('Origin')
            if origin in ["http://localhost:3000", "http://localhost:3001"]:
                response.headers['Access-Control-Allow-Origin'] = origin
                response.headers['Access-Control-Allow-Credentials'] = 'true'
                response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
                response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Session-ID'
            return response

    init_auth(testing)
    init_music(testing)
    init_recommendations(testing)
    init_feedback(testing)
    init_training(testing)

__all__ = [
    'auth_bp',
    'feedback_bp',
    'recommendation_bp',
    'user_bp',
    'music_bp',
    'training_bp',
    'init_routes'
] 