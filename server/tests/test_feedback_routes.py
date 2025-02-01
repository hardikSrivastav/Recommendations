import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from flask import Flask
import json
from application.routes.feedback_routes import feedback_bp, init_db, db_service
from application.models.continuous_learning import UserFeedback

@pytest.fixture(autouse=True)
def setup_db():
    """Initialize database service in testing mode before any tests run."""
    init_db(testing=True)
    yield
    # Reset db_service after each test
    global db_service
    db_service = None

@pytest.fixture
def app():
    """Create a Flask test app."""
    app = Flask(__name__)
    app.register_blueprint(feedback_bp, url_prefix='/api/feedback')
    return app

@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()

@pytest.fixture
def mock_db_service():
    """Create a mock database service."""
    with patch('application.routes.feedback_routes.db_service') as mock:
        # Initialize mock methods
        mock.store_feedback = Mock()
        mock.store_listening_history = Mock()
        mock.get_user_feedback = Mock()
        mock.is_rate_limited = Mock(return_value=False)
        mock.redis = Mock()
        mock.redis.delete = Mock()
        
        # Replace the db_service with our mock
        from application.routes.feedback_routes import db_service
        db_service.store_feedback = mock.store_feedback
        db_service.store_listening_history = mock.store_listening_history
        db_service.get_user_feedback = mock.get_user_feedback
        db_service.is_rate_limited = mock.is_rate_limited
        db_service.redis = mock.redis
        
        yield mock

@pytest.fixture
def mock_learning_manager():
    """Create a mock learning manager."""
    with patch('application.routes.feedback_routes.learning_manager') as mock:
        mock.add_feedback = Mock()
        yield mock

class TestFeedbackRoutes:
    def test_submit_feedback_success(self, client, mock_db_service, mock_learning_manager):
        """Test successful feedback submission."""
        data = {
            'user_id': 'user_123',
            'song_id': 'song_456',
            'interaction_type': 'like',
            'context': {'was_recommended': True, 'recommendation_confidence': 0.85}
        }
        
        response = client.post(
            '/api/feedback/submit',
            json=data,
            content_type='application/json'
        )
        
        assert response.status_code == 201
        response_data = json.loads(response.data)
        assert response_data['message'] == 'Feedback submitted successfully'
        assert response_data['data']['user_id'] == 'user_123'
        assert response_data['data']['song_id'] == 'song_456'
        assert response_data['data']['interaction_type'] == 'like'
        
        # Verify database calls
        assert mock_db_service.store_feedback.called
        assert mock_learning_manager.add_feedback.called
        assert mock_db_service.redis.delete.called
        
    def test_submit_feedback_missing_fields(self, client):
        """Test feedback submission with missing fields."""
        data = {
            'user_id': 'user_123',
            # Missing song_id and interaction_type
        }
        
        response = client.post(
            '/api/feedback/submit',
            json=data,
            content_type='application/json'
        )
        
        assert response.status_code == 400
        response_data = json.loads(response.data)
        assert 'error' in response_data
        assert 'Missing required fields' in response_data['error']
        
    def test_submit_feedback_invalid_interaction(self, client, mock_db_service):
        """Test feedback submission with invalid interaction type."""
        data = {
            'user_id': 'user_123',
            'song_id': 'song_456',
            'interaction_type': 'invalid_type'
        }
        
        response = client.post(
            '/api/feedback/submit',
            json=data,
            content_type='application/json'
        )
        
        assert response.status_code == 400
        response_data = json.loads(response.data)
        assert 'error' in response_data
        assert 'Invalid interaction type' in response_data['error']
        assert 'valid_types' in response_data
        assert set(response_data['valid_types']) == {'play', 'like', 'dislike', 'skip'}
        
    def test_submit_feedback_rate_limited(self, client, mock_db_service):
        """Test feedback submission when rate limited."""
        mock_db_service.is_rate_limited.return_value = True
        
        data = {
            'user_id': 'user_123',
            'song_id': 'song_456',
            'interaction_type': 'like'
        }
        
        response = client.post(
            '/api/feedback/submit',
            json=data,
            content_type='application/json'
        )
        
        assert response.status_code == 429
        response_data = json.loads(response.data)
        assert 'error' in response_data
        assert 'Rate limit exceeded' in response_data['error']
        
    def test_batch_feedback_success(self, client, mock_db_service, mock_learning_manager):
        """Test successful batch feedback submission."""
        data = {
            'feedback': [
                {
                    'user_id': 'user_123',
                    'song_id': 'song_1',
                    'interaction_type': 'play',
                    'context': {'was_recommended': True}
                },
                {
                    'user_id': 'user_123',
                    'song_id': 'song_2',
                    'interaction_type': 'like',
                    'context': {}
                }
            ]
        }
        
        response = client.post(
            '/api/feedback/batch',
            json=data,
            content_type='application/json'
        )
        
        assert response.status_code == 201
        response_data = json.loads(response.data)
        assert len(response_data['results']) == 2
        
        # Verify database calls
        assert mock_db_service.store_feedback.call_count == 2
        assert mock_learning_manager.add_feedback.call_count == 2
        assert mock_db_service.redis.delete.call_count == 2
        
    def test_batch_feedback_invalid_format(self, client):
        """Test batch feedback with invalid format."""
        data = {
            'feedback': 'not_a_list'  # Should be a list
        }
        
        response = client.post(
            '/api/feedback/batch',
            json=data,
            content_type='application/json'
        )
        
        assert response.status_code == 400
        response_data = json.loads(response.data)
        assert 'error' in response_data
        assert 'Invalid request format' in response_data['error']
        assert 'feedback must be a list' in response_data['details']
        
    def test_get_user_feedback_history(self, client, mock_db_service):
        """Test retrieving user feedback history."""
        mock_feedback = [
            {
                'user_id': 'user_123',
                'song_id': 'song_1',
                'interaction_type': 'like',
                'timestamp': datetime.now()
            },
            {
                'user_id': 'user_123',
                'song_id': 'song_2',
                'interaction_type': 'play',
                'timestamp': datetime.now()
            }
        ]
        mock_db_service.get_user_feedback.return_value = mock_feedback
        
        response = client.get('/api/feedback/history/user_123')
        
        assert response.status_code == 200
        response_data = json.loads(response.data)
        assert response_data['user_id'] == 'user_123'
        assert len(response_data['feedback_history']) == 2
        
        # Verify database call
        mock_db_service.get_user_feedback.assert_called_with('user_123', limit=50)
        
    def test_get_user_feedback_history_with_limit(self, client, mock_db_service):
        """Test retrieving user feedback history with custom limit."""
        mock_db_service.get_user_feedback.return_value = []
        
        response = client.get('/api/feedback/history/user_123?limit=10')
        
        assert response.status_code == 200
        
        # Verify database call with custom limit
        mock_db_service.get_user_feedback.assert_called_with('user_123', limit=10)

def test_submit_feedback_success(client, auth_headers, test_user):
    """Test successful feedback submission."""
    data = {
        'song_id': 123,
        'interaction_type': 'like'
    }
    response = client.post('/api/feedback/submit', json=data, headers=auth_headers)
    assert response.status_code == 201
    assert response.json['message'] == 'Feedback submitted successfully'
    assert response.json['data']['user_id'] == str(test_user['_id'])
    assert response.json['data']['song_id'] == data['song_id']

def test_submit_feedback_missing_fields(client, auth_headers):
    """Test feedback submission with missing fields."""
    data = {'song_id': 123}  # Missing interaction_type
    response = client.post('/api/feedback/submit', json=data, headers=auth_headers)
    assert response.status_code == 400
    assert 'Missing required fields' in response.json['error']

def test_submit_feedback_invalid_interaction(client, auth_headers):
    """Test feedback submission with invalid interaction type."""
    data = {
        'song_id': 123,
        'interaction_type': 'invalid_type'
    }
    response = client.post('/api/feedback/submit', json=data, headers=auth_headers)
    assert response.status_code == 400
    assert 'Invalid interaction type' in response.json['error']

def test_batch_feedback_success(client, auth_headers, test_user):
    """Test successful batch feedback submission."""
    data = {
        'feedback': [
            {'song_id': 123, 'interaction_type': 'like'},
            {'song_id': 456, 'interaction_type': 'play'}
        ]
    }
    response = client.post('/api/feedback/batch', json=data, headers=auth_headers)
    assert response.status_code == 201
    assert len(response.json['results']) == 2
    assert all(result['user_id'] == str(test_user['_id']) for result in response.json['results'])

def test_batch_feedback_invalid_format(client, auth_headers):
    """Test batch feedback with invalid format."""
    data = {'feedback': 'not_a_list'}
    response = client.post('/api/feedback/batch', json=data, headers=auth_headers)
    assert response.status_code == 400
    assert 'Invalid request format' in response.json['error']

def test_get_feedback_history_success(client, auth_headers, test_user, db):
    """Test getting user's feedback history."""
    # Add some test feedback first
    feedback_data = {
        'user_id': str(test_user['_id']),
        'song_id': 123,
        'interaction_type': 'like',
        'timestamp': datetime.now()
    }
    db.store_feedback(feedback_data)
    
    response = client.get(f'/api/feedback/history/{test_user["_id"]}', headers=auth_headers)
    assert response.status_code == 200
    assert response.json['user_id'] == str(test_user['_id'])
    assert len(response.json['feedback_history']) > 0

def test_get_feedback_history_unauthorized(client, auth_headers):
    """Test getting another user's feedback history."""
    response = client.get('/api/feedback/history/wrong_user_id', headers=auth_headers)
    assert response.status_code == 403
    assert 'Unauthorized' in response.json['error']

def test_get_user_analytics_success(client, auth_headers, test_user, db):
    """Test getting user analytics."""
    response = client.get(
        f'/api/feedback/analytics/user/{test_user["_id"]}',
        headers=auth_headers
    )
    assert response.status_code == 200
    assert response.json['user_id'] == str(test_user['_id'])
    assert 'analytics' in response.json

def test_get_song_analytics_success(client, auth_headers):
    """Test getting song analytics."""
    response = client.get('/api/feedback/analytics/song/123')
    assert response.status_code == 200
    assert response.json['song_id'] == 123
    assert 'analytics' in response.json

def test_get_genre_analytics_success(client):
    """Test getting genre analytics."""
    response = client.get('/api/feedback/analytics/genre/rock')
    assert response.status_code == 200
    assert response.json['genre'] == 'rock'
    assert 'analytics' in response.json

def test_get_system_analytics_success(client):
    """Test getting system analytics."""
    response = client.get('/api/feedback/analytics/overview')
    assert response.status_code == 200
    assert 'analytics' in response.json

def test_get_recommendation_analytics_success(client):
    """Test getting recommendation analytics."""
    response = client.get('/api/feedback/analytics/recommendations')
    assert response.status_code == 200
    assert 'analytics' in response.json

def test_analytics_invalid_time_range(client, auth_headers, test_user):
    """Test analytics with invalid time range."""
    response = client.get(
        f'/api/feedback/analytics/user/{test_user["_id"]}?range=invalid',
        headers=auth_headers
    )
    assert response.status_code == 400
    assert 'Invalid time range' in response.json['error']

def test_rate_limiting(client, auth_headers, test_user, db):
    """Test rate limiting for feedback submission."""
    # Simulate rate limit exceeded
    db.is_rate_limited = lambda x: True
    
    data = {
        'song_id': 123,
        'interaction_type': 'like'
    }
    response = client.post('/api/feedback/submit', json=data, headers=auth_headers)
    assert response.status_code == 429
    assert 'Rate limit exceeded' in response.json['error'] 