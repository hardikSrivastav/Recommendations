from flask import Blueprint, request, jsonify
from datetime import datetime
from application.models.fma_dataset_processor import FMADatasetProcessor
from application.models.integrator import IntegratedMusicRecommender

music_bp = Blueprint('music', __name__)
fma_processor = FMADatasetProcessor()
recommender = IntegratedMusicRecommender()

@music_bp.route('/search')
def search_songs():
    """Search for songs in the FMA dataset"""
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify([])
    
    # Get the dataset
    dataset = fma_processor.process_dataset()
    
    # Search for songs matching the query
    matches = dataset[
        dataset['track_title'].str.lower().str.contains(query, na=False) |
        dataset['artist_name'].str.lower().str.contains(query, na=False)
    ]
    
    # Format the results
    results = []
    for _, row in matches.head(10).iterrows():
        results.append({
            'id': int(row.name),
            'track_title': row['track_title'],
            'artist_name': row['artist_name'],
            'album_title': row['album_title']
        })
    
    return jsonify(results)

@music_bp.route('/history', methods=['POST'])
def log_listening():
    """Log a song listening event"""
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['user_id', 'song_id']
    if not all(field in data for field in required_fields):
        return jsonify({
            'error': 'Missing required fields',
            'required': required_fields
        }), 400
    
    try:
        # Format listening history data
        listening_data = {
            'user_id': data['user_id'],
            'song_id': data['song_id'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Add optional fields if present
        if 'duration_seconds' in data:
            listening_data['duration_seconds'] = data['duration_seconds']
        if 'completed' in data:
            listening_data['completed'] = data['completed']
        
        # Store listening history (implement storage logic)
        # For now, just return success
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
def get_listening_history(user_id):
    """Get user's listening history"""
    try:
        # Fetch listening history (implement storage/retrieval logic)
        # For now return dummy data
        history = []
        return jsonify(history), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500 