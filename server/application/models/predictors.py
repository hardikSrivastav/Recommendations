from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
import torch
import logging
from collections import defaultdict

@dataclass
class DemographicInput:
    user_id: str
    demographics: Dict[str, Any]
    timestamp: datetime

@dataclass
class PopularityInput:
    time_window: timedelta
    user_context: Optional[Dict[str, Any]] = None

class DemographicPredictor:
    """
    Predicts recommendations based on demographic similarity.
    Uses collaborative filtering among users with similar demographics.
    """
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        
    def _calculate_demographic_similarity(
        self,
        user_demographics: Dict[str, Any],
        other_demographics: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity between two users' demographics.
        Uses weighted matching of demographic features.
        
        Weights:
        - Age group: 0.3 (closer age groups = higher similarity)
        - Location: 0.3 (same location = 1, different = 0)
        - Occupation: 0.2 (same occupation = 1, different = 0)
        - Gender: 0.2 (same gender = 1, different = 0)
        """
        weights = {
            'age_group': 0.3,
            'location': 0.3,
            'occupation': 0.2,
            'gender': 0.2
        }
        
        similarity = 0.0
        
        # Age group similarity (closer = more similar)
        age_diff = abs(user_demographics['age_group_encoded'] - 
                      other_demographics['age_group_encoded'])
        age_sim = 1.0 - (age_diff / 4)  # 4 is max possible difference
        similarity += weights['age_group'] * age_sim
        
        # Direct matches for other features
        for feature in ['location', 'occupation', 'gender']:
            if user_demographics[f'{feature}_encoded'] == other_demographics[f'{feature}_encoded']:
                similarity += weights[feature]
                
        return similarity
        
    def _find_similar_users(
        self,
        user_demographics: Dict[str, Any],
        all_demographics: pd.DataFrame
    ) -> pd.DataFrame:
        """Find users with similar demographics above the threshold."""
        similarities = []
        
        for _, other_user in all_demographics.iterrows():
            if other_user['user_id'] != user_demographics['user_id']:
                similarity = self._calculate_demographic_similarity(
                    user_demographics,
                    other_user
                )
                if similarity >= self.similarity_threshold:
                    similarities.append({
                        'user_id': other_user['user_id'],
                        'similarity': similarity
                    })
                    
        return pd.DataFrame(similarities)
        
    def _analyze_listening_patterns(
        self,
        similar_users: pd.DataFrame,
        listening_history: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Analyze listening patterns of similar users.
        Returns weighted song scores based on user similarity.
        """
        song_scores = defaultdict(float)
        total_similarity = similar_users['similarity'].sum()
        
        for _, similar_user in similar_users.iterrows():
            user_history = listening_history[
                listening_history['user_id'] == similar_user['user_id']
            ]
            
            # Weight each song by user similarity
            weight = similar_user['similarity'] / total_similarity
            for _, interaction in user_history.iterrows():
                song_scores[interaction['song_id']] += weight
                
        return song_scores
        
    async def predict(
        self,
        input_data: DemographicInput,
        listening_history: pd.DataFrame,
        all_demographics: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Generate predictions based on demographic similarity.
        Returns dict of song_id -> score mappings.
        """
        # Find similar users
        similar_users = self._find_similar_users(
            input_data.demographics,
            all_demographics
        )
        
        if similar_users.empty:
            logging.warning(f"No similar users found for user {input_data.user_id}")
            return {}
            
        # Get recommendations from similar users
        song_scores = self._analyze_listening_patterns(
            similar_users,
            listening_history
        )
        
        return song_scores

class PopularityPredictor:
    """
    Predicts recommendations based on song popularity.
    Uses time-weighted interaction counts and optional demographic filtering.
    """
    def __init__(self, max_age_days: int = 30):
        self.max_age_days = max_age_days
        
    def _calculate_time_weight(self, age_days: float) -> float:
        """Calculate exponential decay weight based on age."""
        decay_rate = 0.1  # Slower decay = longer relevance
        return np.exp(-decay_rate * age_days)
        
    def _calculate_popularity_scores(
        self,
        listening_history: pd.DataFrame,
        current_time: datetime
    ) -> Dict[str, float]:
        """
        Calculate time-weighted popularity scores.
        Recent interactions have higher weight.
        """
        song_scores = defaultdict(float)
        
        for _, interaction in listening_history.iterrows():
            age_days = (current_time - interaction['timestamp']).days
            if age_days <= self.max_age_days:
                weight = self._calculate_time_weight(age_days)
                song_scores[interaction['song_id']] += weight
                
        # Normalize scores
        if song_scores:
            max_score = max(song_scores.values())
            song_scores = {
                song_id: score/max_score 
                for song_id, score in song_scores.items()
            }
            
        return song_scores
        
    def _apply_demographic_filter(
        self,
        song_scores: Dict[str, float],
        user_context: Dict[str, Any],
        listening_history: pd.DataFrame,
        demographics: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Filter and adjust scores based on demographic preferences.
        Uses demographic similarity to weight song preferences.
        """
        if not user_context or 'demographics' not in user_context:
            return song_scores
            
        demographic_predictor = DemographicPredictor()
        similar_users = demographic_predictor._find_similar_users(
            user_context['demographics'],
            demographics
        )
        
        if similar_users.empty:
            return song_scores
            
        # Adjust scores based on demographic preferences
        adjusted_scores = {}
        for song_id, base_score in song_scores.items():
            demographic_multiplier = 1.0
            
            # Check if similar users liked this song
            similar_user_interactions = listening_history[
                (listening_history['song_id'] == song_id) &
                (listening_history['user_id'].isin(similar_users['user_id']))
            ]
            
            if not similar_user_interactions.empty:
                demographic_multiplier = 1.2  # Boost score if similar users liked it
                
            adjusted_scores[song_id] = base_score * demographic_multiplier
            
        return adjusted_scores
        
    async def predict(
        self,
        input_data: PopularityInput,
        listening_history: pd.DataFrame,
        demographics: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Generate predictions based on popularity.
        Returns dict of song_id -> score mappings.
        """
        # Calculate base popularity scores
        current_time = datetime.now()
        song_scores = self._calculate_popularity_scores(
            listening_history,
            current_time
        )
        
        # Apply demographic filtering if context available
        if input_data.user_context and demographics is not None:
            song_scores = self._apply_demographic_filter(
                song_scores,
                input_data.user_context,
                listening_history,
                demographics
            )
            
        return song_scores 