import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json
from application.database.database_service import DatabaseService, UserDemographics, SongMetadata

@pytest.fixture
def mock_session():
    """Create a mock SQLAlchemy session."""
    session = Mock()
    session.query = Mock()
    session.commit = Mock()
    session.rollback = Mock()
    session.close = Mock()
    return session

@pytest.fixture
def mock_engine():
    """Create a mock SQLAlchemy engine."""
    engine = Mock()
    engine.dispose = Mock()
    return engine

@pytest.fixture
def mock_mongo_client():
    """Create a mock MongoDB client."""
    client = Mock()
    db = Mock()
    
    # Mock collections
    feedback_collection = Mock()
    history_collection = Mock()
    predictions_collection = Mock()
    
    # Set up collection mocks
    db.feedback = feedback_collection
    db.listening_history = history_collection
    db.predictions = predictions_collection
    
    # Mock find operations
    for collection in [feedback_collection, history_collection, predictions_collection]:
        collection.find = Mock(return_value=collection)
        collection.sort = Mock(return_value=collection)
        collection.limit = Mock(return_value=[])
        collection.insert_one = Mock()
        collection.create_index = Mock()
    
    client.music_db = db
    client.close = Mock()
    return client

@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis = Mock()
    redis.setex = Mock()
    redis.get = Mock()
    redis.incr = Mock()
    redis.expire = Mock()
    redis.delete = Mock()
    redis.close = Mock()
    return redis

@pytest.fixture
def db_service(mock_session, mock_engine, mock_mongo_client, mock_redis):
    """Create a DatabaseService with mocked connections."""
    with patch('application.database.database_service.create_engine') as mock_create_engine, \
         patch('application.database.database_service.sessionmaker') as mock_sessionmaker, \
         patch('application.database.database_service.MongoClient') as mock_mongo, \
         patch('application.database.database_service.redis.from_url') as mock_redis_from_url:
        
        # Set up mocks
        mock_create_engine.return_value = mock_engine
        mock_sessionmaker.return_value = lambda: mock_session
        mock_mongo.return_value = mock_mongo_client
        mock_redis_from_url.return_value = mock_redis
        
        # Create service in testing mode
        service = DatabaseService(testing=True)
        
        # Set up mocked properties and clients
        service._session_maker = lambda: mock_session
        service.mongodb = mock_mongo_client.music_db
        service.redis = mock_redis
        service._mongo_client = mock_mongo_client
        service._pg_engine = mock_engine
        service._redis_client = mock_redis
        
        yield service

class TestDatabaseService:
    def test_store_demographics(self, db_service, mock_session):
        """Test storing user demographics."""
        demo_data = {
            'age': 25,
            'gender': 'M',
            'location': 'US',
            'occupation': 'Engineer'
        }
        
        db_service.store_demographics('user_123', demo_data)
        
        # Verify session operations
        assert mock_session.merge.called
        assert mock_session.commit.called
        assert mock_session.close.called
        
        # Verify correct data
        user_demo = mock_session.merge.call_args[0][0]
        assert isinstance(user_demo, UserDemographics)
        assert user_demo.user_id == 'user_123'
        assert user_demo.age == 25
        assert user_demo.gender == 'M'
        assert user_demo.location == 'US'
        assert user_demo.occupation == 'Engineer'
        
    def test_get_demographics(self, db_service, mock_session):
        """Test retrieving user demographics."""
        # Mock user data
        mock_user = Mock(
            user_id='user_123',
            age=25,
            gender='M',
            location='US',
            occupation='Engineer'
        )
        mock_session.query().get.return_value = mock_user
        
        result = db_service.get_demographics('user_123')
        
        assert result == {
            'user_id': 'user_123',
            'age': 25,
            'gender': 'M',
            'location': 'US',
            'occupation': 'Engineer'
        }
        assert mock_session.close.called
        
    def test_store_feedback(self, db_service):
        """Test storing user feedback."""
        feedback_data = {
            'user_id': 'user_123',
            'song_id': 'song_456',
            'interaction_type': 'like'
        }
        
        db_service.store_feedback(feedback_data)
        
        # Verify MongoDB operation
        assert db_service.mongodb.feedback.insert_one.called
        stored_data = db_service.mongodb.feedback.insert_one.call_args[0][0]
        assert stored_data['user_id'] == 'user_123'
        assert stored_data['song_id'] == 'song_456'
        assert stored_data['interaction_type'] == 'like'
        assert 'timestamp' in stored_data
        
    def test_get_user_feedback(self, db_service):
        """Test retrieving user feedback."""
        mock_feedback = [
            {'user_id': 'user_123', 'song_id': 'song_1', 'interaction_type': 'like'},
            {'user_id': 'user_123', 'song_id': 'song_2', 'interaction_type': 'play'}
        ]
        db_service.mongodb.feedback.find().sort().limit.return_value = mock_feedback
        
        result = db_service.get_user_feedback('user_123', limit=2)
        
        assert result == mock_feedback
        assert db_service.mongodb.feedback.find.called
        
    def test_store_listening_history(self, db_service):
        """Test storing listening history."""
        history_data = {
            'user_id': 'user_123',
            'song_id': 'song_456',
            'was_recommended': True,
            'recommendation_confidence': 0.85
        }
        
        db_service.store_listening_history(history_data)
        
        # Verify MongoDB operation
        assert db_service.mongodb.listening_history.insert_one.called
        stored_data = db_service.mongodb.listening_history.insert_one.call_args[0][0]
        assert stored_data['user_id'] == 'user_123'
        assert stored_data['song_id'] == 'song_456'
        assert stored_data['was_recommended'] is True
        assert stored_data['recommendation_confidence'] == 0.85
        assert 'timestamp' in stored_data
        
    def test_cache_recommendations(self, db_service):
        """Test caching recommendations."""
        recommendations = [
            {'song_id': 'song_1', 'score': 0.9},
            {'song_id': 'song_2', 'score': 0.8}
        ]
        
        db_service.cache_recommendations('user_123', recommendations)
        
        # Verify Redis operation
        assert db_service.redis.setex.called
        key, ttl, value = db_service.redis.setex.call_args[0]
        assert key == 'recommendations:user_123'
        assert ttl == 300  # Default TTL
        assert json.loads(value) == recommendations
        
    def test_get_cached_recommendations(self, db_service):
        """Test retrieving cached recommendations."""
        cached_data = json.dumps([
            {'song_id': 'song_1', 'score': 0.9},
            {'song_id': 'song_2', 'score': 0.8}
        ])
        db_service.redis.get.return_value = cached_data
        
        result = db_service.get_cached_recommendations('user_123')
        
        assert result == json.loads(cached_data)
        assert db_service.redis.get.called
        assert db_service.redis.get.call_args[0][0] == 'recommendations:user_123'
        
    def test_is_rate_limited(self, db_service):
        """Test rate limiting."""
        # Test under limit
        db_service.redis.incr.return_value = 50
        assert not db_service.is_rate_limited('user_123')
        
        # Test over limit
        db_service.redis.incr.return_value = 101
        assert db_service.is_rate_limited('user_123')
        
        # Verify Redis operations
        assert db_service.redis.incr.called
        assert db_service.redis.expire.called
        
    def test_close_connections(self, db_service):
        """Test closing all database connections."""
        db_service.close()
        
        # Verify that all connections are closed
        assert db_service._mongo_client.close.called
        assert db_service._pg_engine.dispose.called
        assert db_service._redis_client.close.called 