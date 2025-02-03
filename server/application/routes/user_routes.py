from flask import Blueprint, request, jsonify
from application.models.integrator import IntegratedMusicRecommender
from application.database.database_service import DatabaseService
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from datetime import datetime, timedelta
from functools import wraps

user_bp = Blueprint('user', __name__)
recommender = IntegratedMusicRecommender()
db_service = None

def init_db(testing: bool = False):
    """Initialize the database service."""
    global db_service
    if db_service is None:
        db_service = DatabaseService(testing=testing)

@user_bp.before_request
def ensure_db():
    """Ensure database service is initialized before each request."""
    if db_service is None:
        init_db()

def session_user(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        session_id = request.headers.get('X-Session-ID')
        if not session_id:
            return jsonify({'error': 'Session ID is missing'}), 401
        try:
            # Create a temporary user object with the session ID
            current_user = {'_id': session_id}
        except Exception as e:
            return jsonify({'error': 'Invalid session'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

@user_bp.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    data = request.get_json()
    
    required_fields = ['username', 'email', 'password']
    if not all(field in data for field in required_fields):
        return jsonify({
            'error': 'Missing required fields',
            'required': required_fields
        }), 400
    
    try:
        # Check if user exists
        if db_service.get_user_by_email(data['email']):
            return jsonify({'error': 'Email already registered'}), 409
            
        # Create user
        user_data = {
            'username': data['username'],
            'email': data['email'],
            'password': generate_password_hash(data['password']),
            'created_at': datetime.now()
        }
        
        user_id = db_service.create_user(user_data)
        
        return jsonify({
            'message': 'User registered successfully',
            'user_id': user_id
        }), 201
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@user_bp.route('/login', methods=['POST'])
def login():
    """Login user"""
    data = request.get_json()
    
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Missing email or password'}), 400
        
    try:
        user = db_service.get_user_by_email(data['email'])
        if not user or not check_password_hash(user['password'], data['password']):
            return jsonify({'error': 'Invalid email or password'}), 401
            
        # Generate token
        token = jwt.encode({
            'user_id': str(user['_id']),
            'exp': datetime.utcnow() + timedelta(days=1)
        }, 'your-secret-key', algorithm='HS256')
        
        return jsonify({
            'token': token,
            'user_id': str(user['_id']),
            'username': user['username']
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@user_bp.route('/preferences', methods=['PUT'])
@session_user
def update_preferences(current_user):
    """Update user preferences"""
    data = request.get_json()
    
    try:
        # Update preferences
        preferences = {
            'favorite_genres': data.get('favorite_genres', []),
            'preferred_artists': data.get('preferred_artists', []),
            'listening_time': data.get('listening_time', 'any'),
            'discovery_mode': data.get('discovery_mode', 'balanced')
        }
        
        db_service.update_user_preferences(str(current_user['_id']), preferences)
        
        return jsonify({
            'message': 'Preferences updated successfully',
            'preferences': preferences
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@user_bp.route('/account', methods=['DELETE'])
@session_user
def delete_account(current_user):
    """Delete user account and all associated data"""
    try:
        user_id = str(current_user['_id'])
        
        # Delete user's demographics data
        db_service.delete_demographics(user_id)
        
        # Delete user's listening history
        db_service.delete_listening_history(user_id)
        
        # Delete user's preferences
        db_service.delete_user_preferences(user_id)
        
        # Finally delete the user account itself
        db_service.delete_user(user_id)
        
        return jsonify({
            'message': 'Account and all associated data deleted successfully'
        }), 200
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in delete_account: {error_details}")  # Log the full error
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@user_bp.route('/demographics', methods=['POST', 'PUT'])
@session_user
def submit_demographics(current_user):
    """Submit or update user demographics data"""
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['age', 'gender', 'location', 'occupation']
    if not all(field in data for field in required_fields):
        return jsonify({
            'error': 'Missing required fields',
            'required': required_fields
        }), 400
    
    try:
        # Format demographics data
        demographics_data = {
            'user_id': str(current_user['_id']),
            'age': int(data['age']),
            'gender': data['gender'],
            'location': data['location'],
            'occupation': data['occupation']
        }
        
        # Store demographics in PostgreSQL
        db_service.store_demographics(str(current_user['_id']), demographics_data)
        
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
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in submit_demographics: {error_details}")  # Log the full error
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@user_bp.route('/profile/<user_id>', methods=['GET'])
@session_user
def get_user_profile(current_user, user_id):
    """Get user profile and preferences"""
    try:
        # Check if requesting own profile or has permission
        if str(current_user['_id']) != user_id:
            return jsonify({'error': 'Unauthorized'}), 403
            
        # Fetch user profile
        profile = db_service.get_user_profile(user_id)
        if not profile:
            return jsonify({'error': 'Profile not found'}), 404
            
        return jsonify(profile), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@user_bp.route('/demographics', methods=['GET'])
@session_user
def get_demographics(current_user):
    """Get user demographics data"""
    try:
        # Fetch demographics data from PostgreSQL
        demographics = db_service.get_demographics(str(current_user['_id']))
        if not demographics:
            return jsonify({
                'error': 'Demographics not found'
            }), 404
            
        return jsonify(demographics), 200
        
    except Exception as e:
        print(f"Error in get_demographics: {str(e)}")  # Add logging
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500 