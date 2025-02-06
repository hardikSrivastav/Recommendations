from flask import Flask, request
from flask_cors import CORS
from application.routes import (
    auth_bp, feedback_bp, recommendation_bp,
    user_bp, music_bp, training_bp, init_routes
)
from application.database.database_service import DatabaseService
from application.models.continuous_learning import ContinuousLearningManager, TrainingConfig
from application.models.recommendation_service import RecommendationService
import asyncio
import threading


def create_app(testing=False):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Configure CORS
    CORS(app, 
        resources={
            r"/api/*": {
                "origins": ["http://localhost:3000", "http://localhost:3001"],
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization", "X-Session-ID"],
                "expose_headers": ["X-Session-ID"],
                "supports_credentials": True,
                "send_wildcard": False
            }
        }
    )

    # Add CORS headers to all responses
    @app.after_request
    def after_request(response):
        origin = request.headers.get('Origin')
        if origin in ["http://localhost:3000", "http://localhost:3001"]:
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Credentials'] = 'true'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Session-ID'
        return response

    # Configure app
    app.config.from_object('config.Config')
    if testing:
        app.config.from_object('config.TestConfig')
    app.config['PROPAGATE_EXCEPTIONS'] = True
    app.config['JSON_SORT_KEYS'] = False  # Preserve JSON order

    # Initialize database
    app.db = DatabaseService(testing=testing)

    # Initialize services
    recommendation_service = RecommendationService()
    training_config = TrainingConfig()
    continuous_learning_manager = ContinuousLearningManager(
        model=recommendation_service.integrated_recommender.recommender,
        config=training_config
    )

    # Start continuous learning in a background thread if not testing
    if not testing:
        def run_continuous_learning():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(continuous_learning_manager.start())
            loop.close()

        continuous_learning_thread = threading.Thread(
            target=run_continuous_learning,
            daemon=True
        )
        continuous_learning_thread.start()

    # Store services in app context
    app.recommendation_service = recommendation_service
    app.continuous_learning_manager = continuous_learning_manager

    # Initialize routes
    init_routes(testing)

    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/api')
    app.register_blueprint(feedback_bp, url_prefix='/api/feedback')
    app.register_blueprint(recommendation_bp, url_prefix='/api/recommendations')
    app.register_blueprint(user_bp, url_prefix='/api/users')
    app.register_blueprint(music_bp, url_prefix='/api/music')
    app.register_blueprint(training_bp, url_prefix='/api/training')

    return app 