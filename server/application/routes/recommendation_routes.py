from flask import Blueprint, request, jsonify, current_app
from application.models.integrator import IntegratedMusicRecommender
from application.models.fma_dataset_processor import FMADatasetProcessor
from application.database.database_service import DatabaseService
from application.utils.session_user import session_user
from application.models.recommendation_service import RecommendationService
from application.services.redis_service import RedisService
import pandas as pd
from functools import wraps
import jwt
import logging
from datetime import datetime
import asyncio

recommendation_bp = Blueprint('recommendations', __name__)
recommender = IntegratedMusicRecommender()
fma_processor = FMADatasetProcessor()
db_service = DatabaseService()
recommendation_service = RecommendationService()
redis_service = RedisService()

def init_db(testing: bool = False):
    """Initialize the database service."""
    global db_service
    if db_service is None:
        db_service = DatabaseService(testing=testing)

@recommendation_bp.before_request
def ensure_db():
    """Ensure database service is initialized before each request."""
    if db_service is None:
        init_db()

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        try:
            token = token.split(' ')[1]  # Remove 'Bearer ' prefix
            data = jwt.decode(token, 'your-secret-key', algorithms=['HS256'])
            current_user = db_service.get_user(data['user_id'])
            if not current_user:
                return jsonify({'error': 'Invalid token'}), 401
        except Exception as e:
            return jsonify({'error': 'Invalid token'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

@recommendation_bp.route('/', methods=['GET'])
def get_recommendations():
    """Get personalized music recommendations for a user."""
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
            
        # Check if we have cached recommendations
        cached_recommendations = db_service.get_cached_recommendations(user_id)
        if cached_recommendations:
            return jsonify({'recommendations': cached_recommendations, 'source': 'cache'})
            
        # Get fresh recommendations
        recommendations = recommendation_service.get_recommendations(user_id)
        
        # Cache the recommendations
        db_service.cache_recommendations(user_id, recommendations)
        
        return jsonify({
            'recommendations': recommendations,
            'source': 'model'
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logging.error(f"Error getting recommendations: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@recommendation_bp.route('/train', methods=['POST'])
async def train_model():
    """Train the recommendation model"""
    try:
        result = await recommendation_service.train_model()
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@recommendation_bp.route('/user/<user_id>', methods=['GET'])
async def get_user_recommendations(user_id):
    """Get recommendations for a specific user"""
    try:
        n = request.args.get('n', default=5, type=int)
        recommendations = await recommendation_service.get_recommendations(user_id, n)
        return jsonify(recommendations), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@recommendation_bp.route('/update', methods=['POST'])
async def update_model():
    """Update the model with new data"""
    try:
        await recommendation_service.update_model()
        return jsonify({"status": "success", "message": "Model updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@recommendation_bp.route('/genre/<genre>', methods=['GET'])
async def get_genre_recommendations(genre):
    """Get recommendations for a specific genre"""
    try:
        n = request.args.get('n', default=10, type=int)
        recommendations = await recommendation_service.get_genre_recommendations(genre, n)
        return jsonify({
            'genre': genre,
            'recommendations': recommendations
        }), 200
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@recommendation_bp.route('/similar/<int:song_id>', methods=['GET'])
async def get_similar_songs(song_id):
    """Get songs similar to a given song"""
    try:
        n = request.args.get('n', default=5, type=int)
        similar_songs = await recommendation_service.get_similar_songs(song_id, n)
        return jsonify({
            'song_id': song_id,
            'similar_songs': similar_songs
        }), 200
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@recommendation_bp.route('/trending', methods=['GET'])
def get_trending_recommendations():
    """Get trending song recommendations"""
    try:
        # Get number of recommendations requested (default 10)
        n_recommendations = int(request.args.get('n', 10))
        
        # Get time range from query params (default 'week')
        time_range = request.args.get('range', 'week')
        valid_ranges = ['day', 'week', 'month']
        if time_range not in valid_ranges:
            return jsonify({
                'error': 'Invalid time range',
                'valid_ranges': valid_ranges
            }), 400
            
        # Check cache first
        cache_key = f"trending:{time_range}"
        cached_trending = db_service.redis.get(cache_key)
        if cached_trending:
            return jsonify(cached_trending), 200
            
        # Get trending songs based on recent popularity and engagement
        trending_songs = asyncio.run(recommender.get_trending_songs(
            time_range=time_range,
            n=n_recommendations
        ))
        
        # Cache trending songs
        db_service.redis.setex(
            cache_key,
            300,  # Cache for 5 minutes
            trending_songs
        )
        
        return jsonify({
            'time_range': time_range,
            'trending_songs': trending_songs
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@recommendation_bp.route('/batch', methods=['POST'])
@token_required
def batch_recommendations(current_user):
    """Get recommendations for multiple users"""
    data = request.get_json()
    
    # Validate request
    if not isinstance(data.get('user_ids', []), list):
        return jsonify({
            'error': 'Invalid request format',
            'details': 'user_ids must be a list'
        }), 400
        
    # Check if user has permission for all requested user_ids
    if not all(str(current_user['_id']) == user_id for user_id in data['user_ids']):
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        n_recommendations = int(data.get('n', 5))
        track_data = fma_processor.process_dataset()
        
        # Get recommendations for each user
        results = {}
        for user_id in data['user_ids']:
            # Get user profile and demographics
            user_profile = db_service.get_user_profile(user_id)
            if not user_profile:
                continue
                
            # Create demographics DataFrame
            demographics_df = pd.DataFrame([{
                'user_id': user_id,
                'age': user_profile.get('age', 25),
                'gender': user_profile.get('gender', 'unknown'),
                'location': user_profile.get('location', 'unknown'),
                'occupation': user_profile.get('occupation', 'unknown')
            }])
            
            # Get recommendations
            recommendations = asyncio.run(recommender.get_recommendations(
                user_id=user_id,
                track_data=track_data,
                demographics_df=demographics_df,
                n=n_recommendations
            ))
            
            results[user_id] = recommendations
            
        return jsonify(results), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@recommendation_bp.route('/personalized', methods=['GET'])
@session_user
def get_personalized_recommendations(current_user):
    """Get personalized recommendations for the current user"""
    try:
        requested_limit = int(request.args.get('limit', 5))
        use_cache = request.args.get('use_cache', 'false').lower() == 'true'
        
        # Calculate buffer size: requested amount + 7
        buffer_limit = requested_limit + 7
        
        user_id = str(current_user['_id'])
        logging.info(f"Getting recommendations for user {user_id} (requested={requested_limit}, buffer={buffer_limit}, use_cache={use_cache})")

        # Check Redis cache if requested
        if use_cache:
            cached = redis_service.get_cached_predictions(user_id)
            if cached:
                logging.info("Returning cached predictions from Redis")
                return jsonify(cached)

        # Ensure model is trained with sufficient data
        if not hasattr(recommender, 'demographics_df') or recommender.demographics_df is None:
            recommender.train_model(num_users=100, interactions_per_user=10, epochs=5)
            logging.info("Model trained with initial synthetic data")
        
        # Get ensemble recommendations with buffer
        recommendations = asyncio.run(recommender.get_ensemble_recommendations(
            user_id=user_id,
            n=buffer_limit  # Request extra songs
        ))
        
        # Add metadata about the buffer
        if recommendations and 'predictions' in recommendations:
            recommendations['metadata'] = {
                'requested_limit': requested_limit,
                'buffer_limit': buffer_limit,
                'total_fetched': len(recommendations['predictions'])
            }
            logging.info(f"Generated {len(recommendations.get('predictions', []))} recommendations (buffer size: {buffer_limit})")
        
        return jsonify(recommendations), 200
        
    except Exception as e:
        logging.error(f"Error getting personalized recommendations: {str(e)}")
        return jsonify({'error': str(e)}), 500

@recommendation_bp.route('/cache', methods=['POST'])
@session_user
def cache_predictions(current_user):
    """Cache predictions for the current user"""
    try:
        user_id = str(current_user['_id'])
        predictions = request.get_json()
        
        logging.info(f"Caching predictions for user {user_id}")
        success = redis_service.cache_predictions(user_id, predictions)
        
        if success:
            return jsonify({'message': 'Predictions cached successfully'}), 200
        else:
            return jsonify({'error': 'Failed to cache predictions'}), 500
            
    except Exception as e:
        logging.error(f"Failed to cache predictions: {str(e)}")
        return jsonify({
            'error': 'Failed to cache predictions',
            'details': str(e)
        }), 500

@recommendation_bp.route('/cache', methods=['DELETE'])
@session_user
def clear_cache(current_user):
    """Clear cached predictions for the current user"""
    try:
        user_id = str(current_user['_id'])
        
        logging.info(f"Clearing cached predictions for user {user_id}")
        success = redis_service.clear_cached_predictions(user_id)
        
        if success:
            return jsonify({'message': 'Cache cleared successfully'}), 200
        else:
            return jsonify({'error': 'Failed to clear cache'}), 500
            
    except Exception as e:
        logging.error(f"Failed to clear cache: {str(e)}")
        return jsonify({
            'error': 'Failed to clear cache',
            'details': str(e)
        }), 500 