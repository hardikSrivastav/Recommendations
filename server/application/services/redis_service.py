import redis
import json
import logging
import os
from typing import Optional, Dict, Any

class RedisService:
    def __init__(self):
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis_client = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            decode_responses=True
        )
        logging.info(f"Initialized Redis connection to {self.redis_host}:{self.redis_port}")

    def cache_predictions(self, user_id: str, predictions: Dict[str, Any]) -> bool:
        """Cache predictions for a user"""
        try:
            key = f"predictions:{user_id}"
            logging.info(f"Caching predictions for user {user_id}")
            self.redis_client.set(key, json.dumps(predictions), ex=3600)  # Cache for 1 hour
            return True
        except Exception as e:
            logging.error(f"Failed to cache predictions: {str(e)}")
            return False

    def get_cached_predictions(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached predictions for a user"""
        try:
            key = f"predictions:{user_id}"
            logging.info(f"Fetching cached predictions for user {user_id}")
            data = self.redis_client.get(key)
            if data:
                logging.info("Found cached predictions")
                return json.loads(data)
            logging.info("No cached predictions found")
            return None
        except Exception as e:
            logging.error(f"Failed to get cached predictions: {str(e)}")
            return None

    def clear_cached_predictions(self, user_id: str) -> bool:
        """Clear cached predictions for a user"""
        try:
            key = f"predictions:{user_id}"
            logging.info(f"Clearing cached predictions for user {user_id}")
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logging.error(f"Failed to clear cached predictions: {str(e)}")
            return False 