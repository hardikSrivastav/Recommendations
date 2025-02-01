from flask import Blueprint, request, jsonify
from datetime import datetime
from bson import json_util
import json
from application.models.continuous_learning import UserFeedback, ContinuousLearningManager
from application.models.integrator import IntegratedMusicRecommender
from application.database.database_service import DatabaseService
from application.utils.session_user import session_user
from application.models.fma_dataset_processor import FMADatasetProcessor

feedback_bp = Blueprint('feedback', __name__)
recommender = IntegratedMusicRecommender()
learning_manager = None  # Will be initialized with the model

# Initialize database service as None - will be lazily initialized
db_service = None

# Initialize FMA processor
fma_processor = FMADatasetProcessor()

def init_db(testing: bool = False):
    """Initialize the database service."""
    global db_service
    if db_service is None:
        db_service = DatabaseService(testing=testing)

def init_learning_manager(model):
    """Initialize the learning manager with the model."""
    global learning_manager
    learning_manager = ContinuousLearningManager(model)

# Do NOT initialize database service here - will be initialized on first request
# or during testing setup

@feedback_bp.before_request
def ensure_db():
    """Ensure database service is initialized before each request."""
    if db_service is None:
        init_db()

@feedback_bp.route('/submit', methods=['POST'])
@session_user
def submit_feedback(current_user):
    """Submit user feedback for a song."""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['song_id', 'interaction_type']  # Removed user_id as it comes from token
        if not all(field in data for field in required_fields):
            return jsonify({
                'error': 'Missing required fields',
                'required': required_fields
            }), 400
            
        # Check rate limiting
        if db_service.is_rate_limited(str(current_user['_id'])):
            return jsonify({
                'error': 'Rate limit exceeded'
            }), 429
            
        # Validate interaction type
        valid_interactions = ['play', 'like', 'dislike', 'skip']
        if data['interaction_type'] not in valid_interactions:
            return jsonify({
                'error': 'Invalid interaction type',
                'valid_types': valid_interactions
            }), 400
            
        # Create feedback object
        feedback = UserFeedback(
            user_id=str(current_user['_id']),
            song_id=data['song_id'],
            interaction_type=data['interaction_type'],
            timestamp=datetime.now(),
            context=data.get('context', {})  # Optional context data
        )
        
        # Store feedback in MongoDB
        feedback_data = {
            'user_id': feedback.user_id,
            'song_id': feedback.song_id,
            'interaction_type': feedback.interaction_type,
            'timestamp': feedback.timestamp,
            'context': feedback.context
        }
        db_service.store_feedback(feedback_data)
        
        # Add feedback to learning manager
        if learning_manager:
            learning_manager.add_feedback(feedback)
            
        # Store listening history if interaction is 'play'
        if feedback.interaction_type == 'play':
            history_data = {
                'user_id': feedback.user_id,
                'song_id': feedback.song_id,
                'timestamp': feedback.timestamp,
                'was_recommended': feedback.context.get('was_recommended', False),
                'recommendation_confidence': feedback.context.get('recommendation_confidence')
            }
            db_service.store_listening_history(history_data)
            
        # Invalidate cached recommendations
        db_service.redis.delete(f"recommendations:{feedback.user_id}")
            
        return jsonify({
            'message': 'Feedback submitted successfully',
            'data': {
                'user_id': feedback.user_id,
                'song_id': feedback.song_id,
                'interaction_type': feedback.interaction_type,
                'timestamp': feedback.timestamp.isoformat()
            }
        }), 201
            
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@feedback_bp.route('/batch', methods=['POST'])
@session_user
def batch_feedback(current_user):
    """Submit multiple feedback entries at once."""
    try:
        data = request.get_json()
        
        if not isinstance(data.get('feedback', []), list):
            return jsonify({
                'error': 'Invalid request format',
                'details': 'feedback must be a list'
            }), 400
            
        results = []
        for entry in data['feedback']:
            # Validate each entry
            if not all(field in entry for field in ['song_id', 'interaction_type']):  # Removed user_id check
                continue
                
            if entry['interaction_type'] not in ['play', 'like', 'dislike', 'skip']:
                continue
                
            # Create feedback object
            feedback = UserFeedback(
                user_id=str(current_user['_id']),  # Use authenticated user's ID
                song_id=entry['song_id'],
                interaction_type=entry['interaction_type'],
                timestamp=datetime.now(),
                context=entry.get('context', {})
            )
            
            # Store feedback in MongoDB
            feedback_data = {
                'user_id': feedback.user_id,
                'song_id': feedback.song_id,
                'interaction_type': feedback.interaction_type,
                'timestamp': feedback.timestamp,
                'context': feedback.context
            }
            db_service.store_feedback(feedback_data)
            
            # Add feedback to learning manager
            if learning_manager:
                learning_manager.add_feedback(feedback)
                
            # Store listening history if interaction is 'play'
            if feedback.interaction_type == 'play':
                history_data = {
                    'user_id': feedback.user_id,
                    'song_id': feedback.song_id,
                    'timestamp': feedback.timestamp,
                    'was_recommended': feedback.context.get('was_recommended', False),
                    'recommendation_confidence': feedback.context.get('recommendation_confidence')
                }
                db_service.store_listening_history(history_data)
                
            # Invalidate cached recommendations
            db_service.redis.delete(f"recommendations:{feedback.user_id}")
                
            results.append({
                'user_id': feedback.user_id,
                'song_id': feedback.song_id,
                'interaction_type': feedback.interaction_type,
                'timestamp': feedback.timestamp.isoformat(),
                'status': 'success'
            })
                
        return jsonify({
            'message': f'Successfully processed {len(results)} feedback entries',
            'results': results
        }), 201
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@feedback_bp.route('/history', methods=['GET'])
@session_user
def get_history(current_user):
    """Get user's listening history with song details"""
    try:
        # Get listening history instead of feedback
        history = db_service.get_listening_history(str(current_user['_id']))
        
        if not history:
            return jsonify({
                'feedback_history': []
            }), 200
        
        # Get the dataset
        dataset = fma_processor.process_dataset()
        if dataset is None:
            return jsonify({
                'error': 'Failed to load song database'
            }), 500
        
        # Transform history data to include song details
        formatted_history = []
        for entry in history:
            try:
                song_id = int(entry['song_id']) if isinstance(entry['song_id'], str) else entry['song_id']
                song_data = dataset.loc[song_id]
                formatted_history.append({
                    'id': song_id,
                    'track_title': song_data['track_title'],
                    'artist_name': song_data['artist_name'],
                    'album_title': song_data['album_title'],
                    'timestamp': entry.get('timestamp')
                })
            except Exception as e:
                print(f"Error processing song {song_id}: {str(e)}")
                continue
        
        return jsonify({
            'feedback_history': formatted_history
        }), 200
        
    except Exception as e:
        print(f"Error in get_history: {str(e)}")
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@feedback_bp.route('/history', methods=['POST'])
@session_user
def add_to_history(current_user):
    """Add a song to user's listening history"""
    data = request.get_json()
    
    if not data or 'song_id' not in data:
        return jsonify({'error': 'Missing song_id'}), 400
        
    try:
        # Convert song_id to int if it's a string
        song_id = int(data['song_id']) if isinstance(data['song_id'], str) else data['song_id']
        
        history_data = {
            'user_id': str(current_user['_id']),
            'song_id': song_id,
            'timestamp': datetime.utcnow(),
            'source': 'manual_add'  # To indicate this was manually added to history
        }
        
        # Store in listening_history instead of feedback
        db_service.store_listening_history(history_data)
        
        # Use json_util.dumps to handle MongoDB types
        return json.loads(json_util.dumps({
            'message': 'Song added to history',
            'data': history_data
        })), 201
        
    except Exception as e:
        print(f"Error in add_to_history: {str(e)}")
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@feedback_bp.route('/history/<int:song_id>', methods=['DELETE'])
@session_user
def remove_from_history(current_user, song_id):
    """Remove a song from user's listening history"""
    try:
        db_service.remove_feedback(str(current_user['_id']), song_id)
        return jsonify({
            'message': 'Song removed from history'
        }), 200
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@feedback_bp.route('/analytics/user/<user_id>', methods=['GET'])
@session_user
def get_user_analytics(current_user, user_id):
    """Get analytics for a specific user's feedback and listening history"""
    try:
        # Check if requesting own analytics or has permission
        if str(current_user['_id']) != user_id:
            return jsonify({'error': 'Unauthorized'}), 403
            
        # Get time range from query params (default 'all')
        time_range = request.args.get('range', 'all')
        valid_ranges = ['day', 'week', 'month', 'year', 'all']
        if time_range not in valid_ranges:
            return jsonify({
                'error': 'Invalid time range',
                'valid_ranges': valid_ranges
            }), 400
            
        # Get user analytics
        analytics = db_service.get_user_analytics(user_id, time_range=time_range)
        
        return jsonify({
            'user_id': user_id,
            'time_range': time_range,
            'analytics': analytics
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@feedback_bp.route('/analytics/song/<int:song_id>', methods=['GET'])
def get_song_analytics(song_id):
    """Get analytics for a specific song's feedback"""
    try:
        # Get time range from query params (default 'all')
        time_range = request.args.get('range', 'all')
        valid_ranges = ['day', 'week', 'month', 'year', 'all']
        if time_range not in valid_ranges:
            return jsonify({
                'error': 'Invalid time range',
                'valid_ranges': valid_ranges
            }), 400
            
        # Get song analytics
        analytics = db_service.get_song_analytics(song_id, time_range=time_range)
        
        # Get song details
        dataset = fma_processor.process_dataset()
        song_data = dataset.loc[song_id]
        if song_data.empty:
            return jsonify({'error': 'Song not found'}), 404
            
        return jsonify({
            'song_id': song_id,
            'track_title': song_data['track_title'],
            'artist_name': song_data['artist_name'],
            'time_range': time_range,
            'analytics': analytics
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@feedback_bp.route('/analytics/genre/<genre>', methods=['GET'])
def get_genre_analytics(genre):
    """Get analytics for a specific genre's feedback"""
    try:
        # Get time range from query params (default 'all')
        time_range = request.args.get('range', 'all')
        valid_ranges = ['day', 'week', 'month', 'year', 'all']
        if time_range not in valid_ranges:
            return jsonify({
                'error': 'Invalid time range',
                'valid_ranges': valid_ranges
            }), 400
            
        # Get genre analytics
        analytics = db_service.get_genre_analytics(genre, time_range=time_range)
        
        return jsonify({
            'genre': genre,
            'time_range': time_range,
            'analytics': analytics
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@feedback_bp.route('/analytics/overview', methods=['GET'])
def get_system_analytics():
    """Get overview analytics for the entire system"""
    try:
        # Get time range from query params (default 'all')
        time_range = request.args.get('range', 'all')
        valid_ranges = ['day', 'week', 'month', 'year', 'all']
        if time_range not in valid_ranges:
            return jsonify({
                'error': 'Invalid time range',
                'valid_ranges': valid_ranges
            }), 400
            
        # Check cache first
        cache_key = f"system_analytics:{time_range}"
        cached_analytics = db_service.redis.get(cache_key)
        if cached_analytics:
            return jsonify(cached_analytics), 200
            
        # Get system analytics
        analytics = db_service.get_system_analytics(time_range=time_range)
        
        # Cache analytics
        db_service.redis.setex(
            cache_key,
            300,  # Cache for 5 minutes
            analytics
        )
        
        return jsonify({
            'time_range': time_range,
            'analytics': analytics
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@feedback_bp.route('/analytics/recommendations', methods=['GET'])
def get_recommendation_analytics():
    """Get analytics for recommendation performance"""
    try:
        # Get time range from query params (default 'all')
        time_range = request.args.get('range', 'all')
        valid_ranges = ['day', 'week', 'month', 'year', 'all']
        if time_range not in valid_ranges:
            return jsonify({
                'error': 'Invalid time range',
                'valid_ranges': valid_ranges
            }), 400
            
        # Check cache first
        cache_key = f"recommendation_analytics:{time_range}"
        cached_analytics = db_service.redis.get(cache_key)
        if cached_analytics:
            return jsonify(cached_analytics), 200
            
        # Get recommendation analytics
        analytics = db_service.get_recommendation_analytics(time_range=time_range)
        
        # Cache analytics
        db_service.redis.setex(
            cache_key,
            300,  # Cache for 5 minutes
            analytics
        )
        
        return jsonify({
            'time_range': time_range,
            'analytics': analytics
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500 