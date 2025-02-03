from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from application.database.database_service import DatabaseService
from application.models.model import RecommenderSystem
from application.models.fma_dataset_processor import FMADatasetProcessor
from application.models.integrator import IntegratedMusicRecommender

class RecommendationService:
    """Service for managing music recommendations using real user data."""
    
    def __init__(self):
        self.integrated_recommender = IntegratedMusicRecommender()
        
    async def train_model(self):
        """Train the recommendation model"""
        try:
            logging.info("Starting model training...")
            
            # Train with minimal data for initial setup
            self.integrated_recommender.train_model(
                num_users=5,
                interactions_per_user=3,
                epochs=3
            )
            
            logging.info("Model training completed successfully")
            return {"status": "success", "message": "Model trained successfully"}
            
        except Exception as e:
            error_msg = f"Error training model: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)
            
    async def get_recommendations(self, user_id: str, n: int = 5):
        """Get recommendations for a user"""
        try:
            recommendations = await self.integrated_recommender.get_recommendations(
                user_id=user_id,
                n=n
            )
            return recommendations
            
        except Exception as e:
            error_msg = f"Error getting recommendations: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)
            
    async def update_model(self) -> None:
        """Update the model with new data."""
        try:
            logging.info("Starting model update...")
            
            # Update model with new data
            await self.train_model()
            
            logging.info("Model update completed successfully")
            
        except Exception as e:
            logging.error(f"Error updating model: {str(e)}")
            raise