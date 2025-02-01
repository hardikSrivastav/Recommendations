from flask import Blueprint, request, jsonify
from datetime import datetime
from application.models.fma_dataset_processor import FMADatasetProcessor
from application.models.integrator import IntegratedMusicRecommender
from application.database.database_service import DatabaseService
from application.services.elasticsearch_service import ElasticsearchService
from functools import wraps
import jwt
from application.utils.session_user import session_user

music_bp = Blueprint('music', __name__)
fma_processor = FMADatasetProcessor()
recommender = IntegratedMusicRecommender()
db_service = None
es_service = ElasticsearchService()

def init_services(testing: bool = False):
    """Initialize database and elasticsearch services."""
    global db_service
    if db_service is None:
        db_service = DatabaseService(testing=testing)
        
    # Initialize Elasticsearch index if needed
    try:
        es_service.create_index()
        # Index songs if index is empty
        if not es_service.es.count(index=es_service.index_name)['count']:
            dataset = fma_processor.process_dataset()
            if dataset is not None:
                es_service.index_songs(dataset)
    except Exception as e:
        print(f"Elasticsearch initialization error: {str(e)}")

@music_bp.before_request
def ensure_services():
    """Ensure all services are initialized before each request."""
    if db_service is None:
        init_services()

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

@music_bp.route('/search')
@session_user
def search_songs(current_user):
    """Search for songs using Elasticsearch"""
    query = request.args.get('q', '').strip()
    limit = int(request.args.get('limit', 5))  # Default to 5 results
    
    if not query:
        return jsonify({
            'results': [],
            'total_results': 0,
            'limit': limit,
            'query': query
        }), 200
        
    try:
        # Search using Elasticsearch
        results, total_hits = es_service.search_songs(query, limit)
        
        return jsonify({
            'query': query,
            'results': results,
            'total_results': total_hits,
            'limit': limit
        }), 200
        
    except Exception as e:
        print(f"Search error: {str(e)}")  # Add logging
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@music_bp.route('/songs/<int:song_id>')
@session_user
def get_song_details(current_user, song_id):
    """Get details for a specific song"""
    try:
        # Get the dataset
        dataset = fma_processor.process_dataset()
        
        # Get song details
        if song_id not in dataset.index:
            return jsonify({'error': 'Song not found'}), 404
            
        song = dataset.loc[song_id]
        
        return jsonify({
            'id': int(song.name),
            'track_title': song['track_title'],
            'artist_name': song['artist_name'],
            'album_title': song['album_title'],
            'track_genres': song.get('track_genres', []),
            'track_date_created': song.get('track_date_created'),
            'track_duration': song.get('track_duration')
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@music_bp.route('/popular')
def get_popular_songs():
    """Get popular songs based on listening history and ratings"""
    try:
        # Get limit from query params (default 10)
        limit = int(request.args.get('limit', 10))
        
        # Get time range from query params (default 'week')
        time_range = request.args.get('range', 'week')
        valid_ranges = ['day', 'week', 'month', 'year', 'all']
        if time_range not in valid_ranges:
            return jsonify({
                'error': 'Invalid time range',
                'valid_ranges': valid_ranges
            }), 400
            
        # Get popular songs from database
        popular_songs = db_service.get_popular_songs(limit=limit, time_range=time_range)
        
        # Get song details for each popular song
        dataset = fma_processor.process_dataset()
        results = []
        for song in popular_songs:
            song_id = song['song_id']
            song_data = dataset.loc[song_id]
            if not song_data.empty:
                results.append({
                    'id': int(song_id),
                    'track_title': song_data['track_title'],
                    'artist_name': song_data['artist_name'],
                    'album_title': song_data['album_title'],
                    'play_count': song['play_count'],
                    'like_count': song['like_count']
                })
                
        return jsonify(results), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@music_bp.route('/genres')
def get_genres():
    """Get list of available genres"""
    try:
        # Get the dataset
        dataset = fma_processor.process_dataset()
        
        # Get unique genres
        genres = dataset['track_genres'].dropna().unique().tolist()
        
        return jsonify(genres), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@music_bp.route('/genres/<genre>')
def get_songs_by_genre(genre):
    """Get songs of a specific genre"""
    try:
        # Get limit from query params (default 20)
        limit = int(request.args.get('limit', 20))
        
        # Get offset for pagination
        offset = int(request.args.get('offset', 0))
        
        # Get the dataset
        dataset = fma_processor.process_dataset()
        
        # Filter songs by genre
        genre_songs = dataset[dataset['track_genres'].str.contains(genre, case=False, na=False)]
        
        # Apply pagination
        paginated_songs = genre_songs.iloc[offset:offset + limit]
        
        # Format results
        results = []
        for _, song in paginated_songs.iterrows():
            results.append({
                'id': int(song.name),
                'track_title': song['track_title'],
                'artist_name': song['artist_name'],
                'album_title': song['album_title']
            })
            
        return jsonify({
            'genre': genre,
            'total': len(genre_songs),
            'offset': offset,
            'limit': limit,
            'songs': results
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@music_bp.route('/history', methods=['POST'])
@token_required
def log_listening(current_user):
    """Log a song listening event"""
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['song_id']
    if not all(field in data for field in required_fields):
        return jsonify({
            'error': 'Missing required fields',
            'required': required_fields
        }), 400
    
    try:
        # Format listening history data
        listening_data = {
            'user_id': str(current_user['_id']),
            'song_id': data['song_id'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Add optional fields if present
        if 'duration_seconds' in data:
            listening_data['duration_seconds'] = data['duration_seconds']
        if 'completed' in data:
            listening_data['completed'] = data['completed']
        
        # Store listening history
        db_service.store_listening_history(listening_data)
        
        return jsonify({
            'message': 'Listening event logged successfully',
            'data': listening_data
        }), 201
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@music_bp.route('/history/<user_id>', methods=['GET'])
@token_required
def get_listening_history(current_user, user_id):
    """Get user's listening history"""
    try:
        # Check if requesting own history or has permission
        if str(current_user['_id']) != user_id:
            return jsonify({'error': 'Unauthorized'}), 403
            
        # Get limit from query params (default 50)
        limit = int(request.args.get('limit', 50))
        
        # Get offset for pagination
        offset = int(request.args.get('offset', 0))
        
        # Get listening history
        history = db_service.get_listening_history(user_id, limit=limit, offset=offset)
        
        # Get song details for each history entry
        dataset = fma_processor.process_dataset()
        results = []
        for entry in history:
            song_id = entry['song_id']
            song_data = dataset.loc[song_id]
            if not song_data.empty:
                results.append({
                    'id': int(song_id),
                    'track_title': song_data['track_title'],
                    'artist_name': song_data['artist_name'],
                    'album_title': song_data['album_title'],
                    'timestamp': entry['timestamp'],
                    'duration_seconds': entry.get('duration_seconds'),
                    'completed': entry.get('completed', True)
                })
                
        return jsonify({
            'user_id': user_id,
            'total': db_service.get_listening_history_count(user_id),
            'offset': offset,
            'limit': limit,
            'history': results
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500 