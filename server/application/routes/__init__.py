from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Import blueprints
    from .user_routes import user_bp
    from .music_routes import music_bp
    from .recommendation_routes import recommendation_bp

    # Register blueprints
    app.register_blueprint(user_bp, url_prefix='/api/users')
    app.register_blueprint(music_bp, url_prefix='/api/music')
    app.register_blueprint(recommendation_bp, url_prefix='/api/recommendations')

    return app 