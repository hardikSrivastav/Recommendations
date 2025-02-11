from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json
from sqlalchemy import create_engine, Column, Integer, String, ARRAY, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pymongo import MongoClient, ASCENDING
import redis
import torch

Base = declarative_base()

# PostgreSQL Models
class UserDemographics(Base):
    __tablename__ = 'user_demographics'
    user_id = Column(String(50), primary_key=True)
    age = Column(Integer, nullable=False)
    gender = Column(String(10), nullable=False)
    location = Column(String(2), nullable=False)
    occupation = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SongMetadata(Base):
    __tablename__ = 'songs'
    song_id = Column(String(50), primary_key=True)
    title = Column(String(200), nullable=False)
    artist = Column(String(200), nullable=False)
    album = Column(String(200))
    duration = Column(Integer)
    genres = Column(ARRAY(String))
    tags = Column(ARRAY(String))
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseService:
    """Service for handling database operations across PostgreSQL, MongoDB, and Redis."""
    
    def __init__(
        self,
        postgres_url: str = "postgresql://music_user:music_password@localhost:5433/music_db",
        mongo_url: str = "mongodb://localhost:27017/music_db",
        redis_url: str = "redis://:music_password@localhost:6380/0",
        testing: bool = False
    ):
        self._postgres_url = postgres_url
        self._mongo_url = mongo_url
        self._redis_url = redis_url
        self._testing = testing
        
        # Initialize as None - will be lazily initialized
        self._pg_engine = None
        self._mongo_client = None
        self._redis_client = None
        self._session_maker = None
        
        if not testing:
            self._initialize_connections()
            
    def _initialize_connections(self):
        """Initialize database connections."""
        if self._testing:
            return  # Skip initialization in testing mode
        
        if self._pg_engine is None:
            self._pg_engine = create_engine(self._postgres_url)
            Base.metadata.create_all(self._pg_engine)
            self._session_maker = sessionmaker(bind=self._pg_engine)
            
        if self._mongo_client is None:
            self._mongo_client = MongoClient(self._mongo_url)
            self.mongodb = self._mongo_client.music_db
            
            # Create MongoDB indexes
            self.mongodb.listening_history.create_index([("user_id", ASCENDING)])
            self.mongodb.listening_history.create_index([("timestamp", ASCENDING)])
            self.mongodb.feedback.create_index([("user_id", ASCENDING)])
            self.mongodb.feedback.create_index([("timestamp", ASCENDING)])
            self.mongodb.predictions.create_index([
                ("user_id", ASCENDING),
                ("timestamp", ASCENDING)
            ])
            
        if self._redis_client is None:
            self._redis_client = redis.from_url(self._redis_url, decode_responses=True)
            
    @property
    def Session(self):
        """Get a session maker, initializing connection if needed."""
        if self._session_maker is None and not self._testing:
            self._initialize_connections()
        return self._session_maker
        
    @property
    def mongodb(self):
        """Get MongoDB database, initializing connection if needed."""
        if not hasattr(self, '_mongodb') and not self._testing:
            self._initialize_connections()
        return getattr(self, '_mongodb', None)
        
    @mongodb.setter
    def mongodb(self, value):
        """Set MongoDB database instance."""
        self._mongodb = value
        
    @property
    def redis(self):
        """Get Redis client, initializing connection if needed."""
        if self._redis_client is None and not self._testing:
            self._initialize_connections()
        return self._redis_client
        
    @redis.setter
    def redis(self, value):
        """Set Redis client instance."""
        self._redis_client = value
        
    # PostgreSQL operations
    def store_demographics(self, user_id: str, demographics: Dict[str, Any]) -> None:
        """Store user demographics in PostgreSQL."""
        session = self.Session()
        try:
            user_demo = UserDemographics(
                user_id=user_id,
                age=demographics['age'],
                gender=demographics['gender'],
                location=demographics['location'],
                occupation=demographics['occupation']
            )
            session.merge(user_demo)  # Use merge for upsert behavior
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
            
    def get_demographics(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user demographics from PostgreSQL."""
        session = self.Session()
        try:
            user_demo = session.query(UserDemographics).get(user_id)
            if user_demo:
                return {
                    'user_id': user_demo.user_id,
                    'age': user_demo.age,
                    'gender': user_demo.gender,
                    'location': user_demo.location,
                    'occupation': user_demo.occupation
                }
            return None
        finally:
            session.close()
            
    # MongoDB operations
    def store_feedback(self, feedback_data: Dict[str, Any]) -> None:
        """Store user feedback in MongoDB."""
        try:
            # Ensure proper data types
            feedback_data = {
                'user_id': str(feedback_data['user_id']),  # Convert to string
                'song_id': int(feedback_data['song_id']),  # Convert to int
                'timestamp': feedback_data['timestamp']
            }
            
            # Store in MongoDB
            self.mongodb.feedback.insert_one(feedback_data)
        except Exception as e:
            logging.error(f"Error storing feedback: {str(e)}")
            raise e

    def get_user_feedback(
        self,
        user_id: str,
        start_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve user feedback from MongoDB."""
        try:
            query = {'user_id': str(user_id)}  # Ensure user_id is string
            if start_time:
                query['timestamp'] = {'$gte': start_time}
                
            return list(
                self.mongodb.feedback.find(query)
                .sort('timestamp', -1)
                .limit(limit)
            )
        except Exception as e:
            logging.error(f"Error getting user feedback: {str(e)}")
            raise e
        
    def store_prediction(self, prediction_data: Dict[str, Any]) -> None:
        """Store model predictions in MongoDB."""
        prediction_data['timestamp'] = datetime.utcnow()
        self.mongodb.predictions.insert_one(prediction_data)
        
    def get_user_predictions(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve recent predictions for a user from MongoDB."""
        return list(
            self.mongodb.predictions.find({'user_id': user_id})
            .sort('timestamp', -1)
            .limit(limit)
        )
        
    def store_listening_history(self, history_data: Dict[str, Any]) -> None:
        """Store listening history in MongoDB."""
        try:
            # Ensure proper data types
            history_data = {
                'user_id': str(history_data['user_id']),
                'song_id': int(history_data['song_id']),
                'timestamp': history_data['timestamp'],
                'source': history_data.get('source', 'unknown'),
                'duration_seconds': history_data.get('duration_seconds'),
                'completed': history_data.get('completed', True)
            }
            
            # Store in MongoDB
            self.mongodb.listening_history.insert_one(history_data)
        except Exception as e:
            logging.error(f"Error storing listening history: {str(e)}")
            raise e
        
    def get_listening_history(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Retrieve user's listening history from MongoDB."""
        try:
            return list(
                self.mongodb.listening_history.find(
                    {'user_id': str(user_id)}
                )
                .sort('timestamp', -1)
                .skip(offset)
                .limit(limit)
            )
        except Exception as e:
            logging.error(f"Error getting listening history: {str(e)}")
            raise e
        
    # Redis operations
    def cache_recommendations(
        self,
        user_id: str,
        recommendations: List[Dict[str, Any]],
        ttl: int = 300  # 5 minutes
    ) -> None:
        """Cache user recommendations in Redis."""
        key = f"recommendations:{user_id}"
        self.redis.setex(
            key,
            ttl,
            json.dumps(recommendations)
        )
        
    def get_cached_recommendations(
        self,
        user_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached recommendations from Redis."""
        key = f"recommendations:{user_id}"
        cached = self.redis.get(key)
        return json.loads(cached) if cached else None
        
    def is_rate_limited(
        self,
        user_id: str,
        limit: int = 100,  # requests per minute
        window: int = 60   # 1 minute
    ) -> bool:
        """Check if user is rate limited."""
        key = f"rate_limit:{user_id}"
        current = self.redis.incr(key)
        # Always set expiry on first hit or refresh it
        self.redis.expire(key, window)
        return current > limit
        
    def close(self):
        """Close all database connections."""
        if self._mongo_client:
            self._mongo_client.close()
        if self._pg_engine:
            self._pg_engine.dispose()
        if self._redis_client:
            self._redis_client.close()

    def create_user(self, user_data: Dict[str, Any]) -> str:
        """Create a new user in MongoDB and return the user ID."""
        if self._testing:
            # For testing, return a mock user ID
            return "test_user_id"
            
        result = self.mongodb.users.insert_one(user_data)
        return str(result.inserted_id)

    def update_user_demographics(self, user_id: str, demographics: Dict[str, Any]) -> None:
        """Store or update user demographics in MongoDB for session-based users."""
        try:
            # Store demographics in MongoDB instead of PostgreSQL for session-based users
            demographics['timestamp'] = datetime.utcnow()
            demographics['user_id'] = user_id
            
            # Use upsert to update if exists, insert if not
            self.mongodb.demographics.update_one(
                {'user_id': user_id},
                {'$set': demographics},
                upsert=True
            )
        except Exception as e:
            logging.error(f"Error updating demographics: {str(e)}")
            raise e

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile including demographics from both PostgreSQL and MongoDB."""
        try:
            # First try PostgreSQL
            postgres_demo = self.get_demographics(user_id)
            if postgres_demo:
                # Convert to the same format as MongoDB data
                return {
                    'user_id': postgres_demo['user_id'],
                    'age': postgres_demo['age'],
                    'age_group': self._get_age_group(postgres_demo['age']),
                    'gender': postgres_demo['gender'],
                    'location': postgres_demo['location'],
                    'occupation': postgres_demo['occupation']
                }

            # If not in PostgreSQL, try MongoDB
            mongo_demo = self.mongodb.demographics.find_one({'user_id': user_id})
            if mongo_demo:
                # Remove MongoDB's _id field
                mongo_demo.pop('_id', None)
                return mongo_demo

            return None

        except Exception as e:
            logging.error(f"Error getting user profile: {str(e)}")
            return None 
        
    def delete_demographics(self, user_id: str) -> None:
        """Delete user demographics from both PostgreSQL and MongoDB."""
        try:
            # Delete from PostgreSQL
            session = self.Session()
            try:
                demo = session.query(UserDemographics).get(user_id)
                if demo:
                    session.delete(demo)
                    session.commit()
            finally:
                session.close()

            # Delete from MongoDB
            self.mongodb.demographics.delete_one({'user_id': user_id})
            
            logging.info(f"Deleted demographics for user {user_id}")
        except Exception as e:
            logging.error(f"Error deleting demographics: {str(e)}")
            raise

    def delete_listening_history(self, user_id: str) -> None:
        """Delete user's listening history from MongoDB."""
        try:
            result = self.mongodb.listening_history.delete_many({'user_id': user_id})
            logging.info(f"Deleted {result.deleted_count} listening history records for user {user_id}")
        except Exception as e:
            logging.error(f"Error deleting listening history: {str(e)}")
            raise

    def delete_user_preferences(self, user_id: str) -> None:
        """Delete user preferences from MongoDB."""
        try:
            # Delete from preferences collection
            self.mongodb.preferences.delete_one({'user_id': user_id})
            # Delete from feedback collection
            self.mongodb.feedback.delete_many({'user_id': user_id})
            # Delete from predictions collection
            self.mongodb.predictions.delete_many({'user_id': user_id})
            
            logging.info(f"Deleted preferences and related data for user {user_id}")
        except Exception as e:
            logging.error(f"Error deleting user preferences: {str(e)}")
            raise

    def delete_user(self, user_id: str) -> None:
        """Delete user account from MongoDB."""
        try:
            # Delete the user document
            self.mongodb.users.delete_one({'_id': user_id})
            
            # Clear any cached data in Redis
            pattern = f"*:{user_id}"
            for key in self.redis.scan_iter(pattern):
                self.redis.delete(key)
                
            logging.info(f"Deleted user account and cleared cache for user {user_id}")
        except Exception as e:
            logging.error(f"Error deleting user account: {str(e)}")
            raise

    def _get_age_group(self, age: int) -> str:
        """Convert age to age group."""
        if age < 18:
            return '<18'
        elif age < 25:
            return '18-25'
        elif age < 35:
            return '26-35'
        elif age < 50:
            return '36-50'
        else:
            return '50+'

    def remove_song(self, user_id: str, song_id: int, timestamp: Optional[datetime] = None) -> None:
        """Remove feedback for a specific song from MongoDB.
        
        Args:
            user_id: The ID of the user
            song_id: The ID of the song
            timestamp: Optional timestamp to identify a specific feedback instance.
                      If None, removes the most recent feedback entry.
        """
        try:
            query = {'user_id': user_id, 'song_id': song_id}
            if timestamp:
                query['timestamp'] = timestamp
                result = self.mongodb.listening_history.delete_one(query)
            else:
                # If no timestamp provided, delete the most recent feedback
                feedback = self.mongodb.listening_history.find_one(
                    query,
                    sort=[('timestamp', -1)]  # Sort by timestamp descending
                )
                if feedback:
                    result = self.mongodb.listening_history.delete_one({'_id': feedback['_id']})
                else:
                    result = None

            deleted = result.deleted_count if result else 0
            logging.info(f"Deleted {deleted} feedback entry for user {user_id} and song {song_id}")
        except Exception as e:
            logging.error(f"Error removing feedback: {str(e)}")
            raise

    def anonymize_user_data(self, user_id: str) -> str:
        """Anonymize user data instead of deleting it to preserve training data.
        Returns the new session user ID."""
        new_user_id = None
        try:
            # Create a new session user first
            new_user_id = self.create_session_user()
            
            # Anonymize MongoDB data first (this is critical)
            # Update listening history to remove identifiable info but keep song interactions
            self.mongodb.listening_history.update_many(
                {'user_id': user_id},
                {'$set': {'user_id': f'anon_{user_id}', 'anonymous': True}}
            )
            
            # Update feedback to preserve ratings but remove user identity
            self.mongodb.feedback.update_many(
                {'user_id': user_id},
                {'$set': {'user_id': f'anon_{user_id}', 'anonymous': True}}
            )
            
            # Update predictions to mark as anonymous
            self.mongodb.predictions.update_many(
                {'user_id': user_id},
                {'$set': {'user_id': f'anon_{user_id}', 'anonymous': True}}
            )
            
            # Try to clear Redis cache if available
            if self.redis:
                try:
                    pattern = f"*:{user_id}"
                    keys = self.redis.keys(pattern)
                    if keys:
                        self.redis.delete(*keys)
                    logging.info(f"Cleared Redis cache for user {user_id}")
                except Exception as redis_error:
                    logging.warning(f"Redis cache clearing failed: {str(redis_error)}")
                    # Continue with the process even if Redis fails
            
            logging.info(f"Successfully anonymized data for user {user_id}")
            return new_user_id
        
        except Exception as e:
            logging.error(f"Error anonymizing user data: {str(e)}")
            # If we created a new user but failed later, try to clean it up
            if new_user_id:
                try:
                    self.mongodb.users.delete_one({'_id': new_user_id})
                except Exception as cleanup_error:
                    logging.error(f"Failed to clean up new user after error: {str(cleanup_error)}")
            raise

    def create_session_user(self) -> str:
        """Create a new session user and return their ID."""
        try:
            # Create a new user document with minimal data
            user_data = {
                'created_at': datetime.utcnow(),
                'type': 'session',
                'status': 'active'
            }
            result = self.mongodb.users.insert_one(user_data)
            new_user_id = str(result.inserted_id)
            
            logging.info(f"Created new session user with ID: {new_user_id}")
            return new_user_id
        
        except Exception as e:
            logging.error(f"Error creating session user: {str(e)}")
            raise