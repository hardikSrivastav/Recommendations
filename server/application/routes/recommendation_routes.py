from flask import Blueprint, request, jsonify
from application.models.integrator import IntegratedMusicRecommender
from application.models.fma_dataset_processor import FMADatasetProcessor
from application.database.database_service import DatabaseService
from application.utils.session_user import session_user
import pandas as pd
from functools import wraps
import jwt
from datetime import datetime

recommendation_bp = Blueprint('recommendation', __name__)
recommender = IntegratedMusicRecommender()
fma_processor = FMADatasetProcessor()
db_service = None

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
@session_user
def get_recommendations(current_user):
    """Get personalized song recommendations"""
    try:
        # Get number of recommendations requested (default 5)
        n_recommendations = int(request.args.get('n', 5))
        
        # Get user profile and demographics
        user_profile = db_service.get_user_profile(str(current_user['_id']))
        if not user_profile:
            return jsonify({'error': 'User profile not found'}), 404
            
        # Create demographics DataFrame
        demographics_df = pd.DataFrame([{
            'user_id': str(current_user['_id']),
            'age': user_profile.get('age', 25),
            'gender': user_profile.get('gender', 'unknown'),
            'location': user_profile.get('location', 'unknown'),
            'occupation': user_profile.get('occupation', 'unknown')
        }])
        
        # Get recommendations
        recommendations = recommender.get_recommendations(
            user_id=str(current_user['_id']),
            demographics_df=demographics_df,
            n=n_recommendations
        )
        
        return jsonify({
            'recommendations': recommendations
        }), 200
        
    except ValueError as e:
        return jsonify({
            'error': 'Invalid request',
            'details': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@recommendation_bp.route('/genre/<genre>', methods=['GET'])
async def get_genre_recommendations(genre):
    """Get recommendations for a specific genre"""
    try:
        # Get number of recommendations requested (default 10)
        n_recommendations = int(request.args.get('n', 10))
        
        # Get the dataset
        track_data = fma_processor.process_dataset()
        
        # Filter songs by genre
        genre_songs = track_data[track_data['track_genres'].str.contains(genre, case=False, na=False)]
        
        if genre_songs.empty:
            return jsonify({
                'error': 'No songs found for this genre'
            }), 404
            
        # Get popular songs in this genre based on play count and ratings
        popular_genre_songs = await recommender.get_genre_recommendations(
            genre=genre,
            track_data=genre_songs,
            n=n_recommendations
        )
        
        return jsonify({
            'genre': genre,
            'recommendations': popular_genre_songs
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
        # Get number of recommendations requested (default 5)
        n_recommendations = int(request.args.get('n', 5))
        
        # Get the dataset
        track_data = fma_processor.process_dataset()
        
        # Check if song exists
        if song_id not in track_data.index:
            return jsonify({'error': 'Song not found'}), 404
            
        # Get similar songs based on audio features and metadata
        similar_songs = await recommender.get_similar_songs(
            song_id=song_id,
            track_data=track_data,
            n=n_recommendations
        )
        
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
async def get_trending_recommendations():
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
        trending_songs = await recommender.get_trending_songs(
            time_range=time_range,
            n=n_recommendations
        )
        
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
async def batch_recommendations(current_user):
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
            recommendations = await recommender.get_recommendations(
                user_id=user_id,
                track_data=track_data,
                demographics_df=demographics_df,
                n=n_recommendations
            )
            
            results[user_id] = recommendations
            
        return jsonify(results), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500 