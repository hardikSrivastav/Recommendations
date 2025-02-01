from functools import wraps
from flask import request, jsonify

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