from flask import Blueprint, request, jsonify
from application.models.integrator import IntegratedMusicRecommender

user_bp = Blueprint('user', __name__)
recommender = IntegratedMusicRecommender()

@user_bp.route('/demographics', methods=['POST'])
def submit_demographics():
    """Submit user demographics data"""
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['user_id', 'age', 'gender', 'location', 'occupation']
    if not all(field in data for field in required_fields):
        return jsonify({
            'error': 'Missing required fields',
            'required': required_fields
        }), 400
    
    try:
        # Format demographics data for the model
        demographics_data = {
            'user_id': data['user_id'],
            'age': int(data['age']),
            'gender': data['gender'],
            'location': data['location'],
            'occupation': data['occupation']
        }
        
        # Store demographics (you'll need to implement storage logic)
        # For now, we'll just return success
        return jsonify({
            'message': 'Demographics submitted successfully',
            'data': demographics_data
        }), 201
        
    except ValueError as e:
        return jsonify({
            'error': 'Invalid data format',
            'details': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@user_bp.route('/profile/<user_id>', methods=['GET'])
def get_user_profile(user_id):
    """Get user profile and preferences"""
    try:
        # Fetch user profile (implement storage/retrieval logic)
        # For now return dummy data
        profile = {
            'user_id': user_id,
            'total_songs_played': 0,
            'favorite_genres': [],
            'listening_time': 0
        }
        return jsonify(profile), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500 