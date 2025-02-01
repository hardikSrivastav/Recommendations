from flask import Blueprint, request, jsonify
from application.database.database_service import DatabaseService
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from datetime import datetime, timedelta
from functools import wraps

auth_bp = Blueprint('auth', __name__)
db_service = None

def init_db(testing: bool = False):
    """Initialize the database service."""
    global db_service
    if db_service is None:
        db_service = DatabaseService(testing=testing)

@auth_bp.before_request
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
            if not token.startswith('Bearer '):
                return jsonify({'error': 'Invalid token format'}), 401
            token = token.split(' ')[1]  # Remove 'Bearer ' prefix
            data = jwt.decode(token, 'your-secret-key', algorithms=['HS256'])
            current_user = db_service.get_user(data['user_id'])
            if not current_user:
                return jsonify({'error': 'Invalid token'}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except Exception as e:
            return jsonify({'error': 'Invalid token'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

@auth_bp.route('/profile/<user_id>', methods=['GET'])
@token_required
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