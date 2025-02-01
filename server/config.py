import os
from datetime import timedelta

class Config:
    """Base configuration."""
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/music')
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    DATASET_PATH = os.getenv('DATASET_PATH', '../data/fma/tracks.csv')

class TestConfig(Config):
    """Test configuration."""
    TESTING = True
    MONGODB_URI = 'mongodb://localhost:27017/music_test'
    JWT_SECRET_KEY = 'test-jwt-secret-key'
    DATASET_PATH = '../data/fma/test_tracks.csv' 