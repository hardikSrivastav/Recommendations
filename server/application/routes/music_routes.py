from flask import Blueprint, request, jsonify
from datetime import datetime
from application.models.fma_dataset_processor import FMADatasetProcessor
from application.models.integrator import IntegratedMusicRecommender
from application.database.database_service import DatabaseService
from application.database.postgres_service import PostgresService
from application.services.elasticsearch_service import ElasticsearchService
from functools import wraps
import jwt
from application.utils.session_user import session_user
import json
from bson import json_util
import logging

music_bp = Blueprint('music', __name__)
fma_processor = FMADatasetProcessor()
recommender = IntegratedMusicRecommender()
db_service = None
pg_service = None
es_service = ElasticsearchService()

def init_services(testing: bool = False):
    """Initialize database and elasticsearch services."""
    global db_service, pg_service
    if db_service is None:
        db_service = DatabaseService(testing=testing)
    if pg_service is None:
        pg_service = PostgresService(testing=testing)
        
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
    if db_service is None or pg_service is None:
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

@music_bp.route('/songs/<int:song_id>', methods=['GET', 'OPTIONS'])
def get_song_details(song_id):
    """Get details for a specific song"""
    # Handle OPTIONS request for CORS
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        return response

    try:
        logging.info(f"Fetching details for song {song_id}")
        # Get song details from PostgreSQL
        tracks = pg_service.get_tracks([song_id])
        
        if tracks is None or tracks.empty:
            logging.warning(f"Song {song_id} not found in database")
            return jsonify({
                'error': 'Song not found',
                'song_id': song_id
            }), 404
            
        # Get the song details (first row since we queried by ID)
        try:
            song = tracks.iloc[0]
        except IndexError:
            logging.error(f"Failed to access song data for ID {song_id}")
            return jsonify({
                'error': 'Failed to access song data',
                'song_id': song_id
            }), 500
        
        try:
            response_data = {
                'id': int(song.name),  # Index is the song ID
                'track_title': str(song['track_title']),
                'artist_name': str(song['artist_name']),
                'album_title': str(song['album_title']),
                'track_genres': eval(song['track_genres']) if isinstance(song['track_genres'], str) else [],
                'track_date_created': None,  # Not available in our dataset
                'track_duration': float(song['duration']),
                'track_tags': eval(song['track_tags']) if isinstance(song['track_tags'], str) else []
            }
        except Exception as e:
            logging.error(f"Error formatting song data for ID {song_id}: {str(e)}")
            return jsonify({
                'error': 'Error formatting song data',
                'details': str(e),
                'song_id': song_id
            }), 500
        
        logging.info(f"Successfully retrieved details for song {song_id}")
        return jsonify(response_data), 200
            
    except Exception as e:
        logging.error(f"Error retrieving song {song_id}: {str(e)}")
        return jsonify({
            'error': 'Server error',
            'details': str(e),
            'song_id': song_id
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
        song_ids = [song['song_id'] for song in popular_songs]
        tracks = pg_service.get_tracks(song_ids)
        
        results = []
        for song in popular_songs:
            song_id = song['song_id']
            if song_id in tracks.index:
                song_data = tracks.loc[song_id]
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
        # Get unique genres from PostgreSQL
        tracks = pg_service.get_tracks()
        genres = []
        for track_genres in tracks['track_genres']:
            try:
                genre_list = eval(track_genres)
                genres.extend(genre_list)
            except:
                continue
        
        # Get unique genres
        unique_genres = list(set(genres))
        
        return jsonify(unique_genres), 200
        
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
        
        # Get tracks from PostgreSQL
        tracks = pg_service.get_tracks()
        
        # Filter songs by genre
        genre_songs = tracks[tracks['track_genres'].apply(
            lambda x: genre.lower() in [g.lower() for g in eval(x)]
        )]
        
        # Apply pagination
        paginated_songs = genre_songs.iloc[offset:offset + limit]
        
        # Format results
        results = []
        for idx, song in paginated_songs.iterrows():
            results.append({
                'id': int(idx),
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

@music_bp.route('/history', methods=['GET'])
@session_user
def get_history(current_user):
    """Get user's listening history with song details"""
    try:
        # Get listening history
        history = db_service.get_listening_history(str(current_user['_id']))
        
        if not history:
            return jsonify({
                'feedback_history': []
            }), 200
        
        # Get song IDs from history
        song_ids = [entry['song_id'] for entry in history]
        
        # Get tracks from PostgreSQL
        tracks = pg_service.get_tracks(song_ids)
        
        # Transform history data to include song details
        formatted_history = []
        for entry in history:
            try:
                song_id = int(entry['song_id']) if isinstance(entry['song_id'], str) else entry['song_id']
                if song_id in tracks.index:
                    song_data = tracks.loc[song_id]
                    formatted_history.append({
                        'id': song_id,
                        'track_title': song_data['track_title'],
                        'artist_name': song_data['artist_name'],
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

@music_bp.route('/history', methods=['POST'])
@session_user
def add_to_history(current_user):
    """Add a song to user's listening history"""
    try:
        data = request.get_json()
        logging.info(f"Received add to history request with data: {data}")
        
        if not data:
            logging.error("No data received in request")
            return jsonify({'error': 'No data received'}), 400
            
        if 'song_id' not in data:
            logging.error(f"Missing song_id in data: {data}")
            return jsonify({'error': 'Missing song_id'}), 400
        
        # Convert song_id to int if it's a string
        try:
            song_id = int(data['song_id']) if isinstance(data['song_id'], str) else data['song_id']
        except (ValueError, TypeError) as e:
            logging.error(f"Invalid song_id format: {data['song_id']}, error: {str(e)}")
            return jsonify({'error': 'Invalid song_id format'}), 400
        
        # Verify song exists in PostgreSQL
        tracks = pg_service.get_tracks([song_id])
        if tracks.empty:
            logging.error(f"Song not found in database: {song_id}")
            return jsonify({'error': 'Invalid song_id - song not found'}), 400
        
        history_data = {
            'user_id': str(current_user['_id']),
            'song_id': song_id,
            'timestamp': datetime.utcnow(),
            'source': 'manual_add'  # To indicate this was manually added to history
        }
        
        # Store in listening_history
        db_service.store_listening_history(history_data)
        logging.info(f"Successfully added song {song_id} to history for user {current_user['_id']}")
        
        # Use json_util.dumps to handle MongoDB types
        return json.loads(json_util.dumps({
            'message': 'Song added to history',
            'data': history_data
        })), 201
        
    except Exception as e:
        logging.error(f"Error in add_to_history: {str(e)}")
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@music_bp.route('/history/<int:song_id>', methods=['DELETE'])
@session_user
def remove_from_history(current_user, song_id):
    """Remove a song from user's listening history"""
    try:
        db_service.remove_song(str(current_user['_id']), song_id)
        return jsonify({
            'message': 'Song removed from history'
        }), 200
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500 