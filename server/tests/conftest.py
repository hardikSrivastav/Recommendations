import pytest
import jwt
from datetime import datetime, timedelta
from application import create_app
from application.database.database_service import DatabaseService
from application.models.continuous_learning import ContinuousLearningManager
from application.models.fma_dataset_processor import FMADatasetProcessor
from unittest.mock import Mock, patch

@pytest.fixture
def app():
    """Create and configure a test Flask application instance."""
    app = create_app(testing=True)
    app.config.update({
        'TESTING': True,
        'SECRET_KEY': 'test-secret-key'
    })
    return app

@pytest.fixture
def client(app):
    """Create a test client for the Flask application."""
    return app.test_client()

@pytest.fixture
def db():
    """Create a test database service instance with mocked connections."""
    with patch('application.database.database_service.MongoClient') as mock_mongo, \
         patch('application.database.database_service.redis.from_url') as mock_redis, \
         patch('application.database.database_service.create_engine') as mock_engine:
        
        # Set up mock MongoDB
        mock_db = Mock()
        mock_db.users = Mock()
        mock_db.feedback = Mock()
        mock_db.listening_history = Mock()
        mock_db.predictions = Mock()
        
        # Set up collections
        for collection in [mock_db.users, mock_db.feedback, mock_db.listening_history, mock_db.predictions]:
            collection.insert_one = Mock(return_value=Mock(inserted_id='test_id'))
            collection.find = Mock(return_value=Mock(
                sort=Mock(return_value=Mock(
                    limit=Mock(return_value=[])
                ))
            ))
            collection.create_index = Mock()
        
        mock_mongo.return_value = Mock(music_db=mock_db)
        mock_redis.return_value = Mock()
        mock_engine.return_value = Mock()
        
        db_service = DatabaseService(testing=True)
        db_service.mongodb = mock_db
        return db_service

@pytest.fixture
def fma_processor():
    """Create an FMA dataset processor instance."""
    return FMADatasetProcessor()

@pytest.fixture
def test_user(db):
    """Create a test user and return their details."""
    user_data = {
        'username': 'testuser',
        'email': 'test@example.com',
        'password': 'hashed_password',
        'created_at': datetime.now()
    }
    user_id = db.create_user(user_data)
    user_data['_id'] = user_id
    return user_data

@pytest.fixture
def auth_token(test_user):
    """Generate a valid authentication token for the test user."""
    token = jwt.encode({
        'user_id': str(test_user['_id']),
        'exp': datetime.utcnow() + timedelta(days=1)
    }, 'your-secret-key', algorithm='HS256')
    return token

@pytest.fixture
def auth_headers(auth_token):
    """Create headers with authentication token."""
    return {'Authorization': f'Bearer {auth_token}'}

@pytest.fixture
def expired_token(test_user):
    """Generate an expired authentication token."""
    token = jwt.encode({
        'user_id': str(test_user['_id']),
        'exp': datetime.utcnow() - timedelta(days=1)
    }, 'your-secret-key', algorithm='HS256')
    return token

@pytest.fixture
def invalid_token():
    """Generate an invalid authentication token."""
    return 'invalid.token.string' 