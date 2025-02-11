from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
import torch
import logging
from collections import defaultdict
from application.database.postgres_service import PostgresService

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
    def __init__(self, similarity_threshold: float = 0.2):  # Lower threshold for sparse data
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
            'age': 0.3,
            'location': 0.3,
            'occupation': 0.2,
            'gender': 0.2
        }
        
        similarity = 0.0
        
        # Age similarity - handle both age and age_group
        try:
            # Get ages, defaulting to 25 if not available
            user_age = int(user_demographics.get('age', 25))
            other_age = int(other_demographics.get('age', 25))
            
            # Calculate age difference and normalize
            age_diff = abs(user_age - other_age)
            max_age_diff = 50  # Maximum age difference to consider
            age_sim = max(0.1, 1.0 - (age_diff / max_age_diff))
            similarity += weights['age'] * age_sim
            
            logging.debug(f"Age similarity: {age_sim} (user_age={user_age}, other_age={other_age})")
        except (ValueError, TypeError) as e:
            logging.warning(f"Age comparison failed: {str(e)}")
            similarity += weights['age'] * 0.1
        
        # Direct matches for other features with partial matching
        for feature in ['location', 'occupation', 'gender']:
            try:
                user_value = str(user_demographics.get(feature, '')).upper()
                other_value = str(other_demographics.get(feature, '')).upper()
                
                if user_value and other_value:
                    if user_value == other_value:
                        similarity += weights[feature]
                        logging.debug(f"{feature} exact match: {user_value}")
                    elif feature == 'location' and len(user_value) >= 2 and len(other_value) >= 2:
                        # For location, check region similarity
                        if user_value[:2] == other_value[:2]:
                            similarity += weights[feature] * 0.8
                            logging.debug(f"{feature} region match: {user_value[:2]}")
                        else:
                            similarity += weights[feature] * 0.1
                    else:
                        # Partial match for non-exact matches
                        similarity += weights[feature] * 0.2
                        logging.debug(f"{feature} partial match: {user_value} vs {other_value}")
                else:
                    # If either value is missing, give small similarity
                    similarity += weights[feature] * 0.1
            except (AttributeError, TypeError) as e:
                logging.warning(f"{feature} comparison failed: {str(e)}")
                similarity += weights[feature] * 0.1
        
        logging.debug(f"Final similarity: {similarity} between users {user_demographics.get('user_id')} and {other_demographics.get('user_id')}")
        return similarity
        
    def _find_similar_users(
        self,
        user_demographics: Dict[str, Any],
        all_demographics: pd.DataFrame
    ) -> pd.DataFrame:
        """Find users with similar demographics above the threshold."""
        similarities = []
        
        # If no other users, return empty DataFrame
        if all_demographics is None or all_demographics.empty:
            logging.warning("No demographics data available for comparison")
            return pd.DataFrame(columns=['user_id', 'similarity'])
        
        current_user_id = user_demographics.get('user_id')
        # Remove 'anon_' prefix if present for comparison
        current_user_base_id = current_user_id.replace('anon_', '')
        
        # Filter out the current user and their anonymized version
        other_users = all_demographics[
            ~all_demographics['user_id'].isin([
                current_user_base_id,
                f'anon_{current_user_base_id}'
            ])
        ]
        
        if other_users.empty:
            logging.warning(f"No other users found to compare with user {current_user_id}")
            return pd.DataFrame(columns=['user_id', 'similarity'])
        
        logging.info(f"Finding similar users for user {current_user_id} among {len(other_users)} other users")
        logging.info(f"User demographics: {user_demographics}")
        
        for _, other_user in other_users.iterrows():
            # Remove 'anon_' prefix from other user for comparison
            other_user_dict = other_user.to_dict()
            other_user_dict['user_id'] = other_user_dict['user_id'].replace('anon_', '')
            
            similarity = self._calculate_demographic_similarity(
                user_demographics,
                other_user_dict
            )
            # Log all similarities, not just those above threshold
            logging.debug(f"Similarity with user {other_user['user_id']}: {similarity}")
            if similarity >= self.similarity_threshold:
                similarities.append({
                    'user_id': other_user['user_id'],
                    'similarity': similarity
                })
                logging.info(f"Found similar user: {other_user['user_id']} with similarity {similarity}")
                
        result_df = pd.DataFrame(similarities)
        logging.info(f"Found {len(result_df)} similar users above threshold {self.similarity_threshold}")
        if result_df.empty:
            logging.warning("No users found above similarity threshold - consider lowering threshold")
        return result_df
        
    def _analyze_listening_patterns(
        self,
        similar_users: pd.DataFrame,
        listening_history: pd.DataFrame,
        current_user_id: str = None
    ) -> Dict[str, float]:
        """Analyze listening patterns of similar users to generate song scores."""
        if similar_users.empty:
            logging.warning("No similar users found")
            return {}

        # Get all songs from PostgreSQL
        pg_service = PostgresService()
        all_tracks = pg_service.get_tracks()
        total_songs = len(all_tracks)
        logging.info(f"Total songs in database: {total_songs}")

        # Get current user's songs to exclude
        current_user_songs = set()
        if current_user_id:
            current_user_history = listening_history[listening_history['user_id'] == current_user_id]
            current_user_songs = set(str(song_id) for song_id in current_user_history['song_id'])
            logging.info(f"Found {len(current_user_songs)} songs in current user's history to exclude")

        # Initialize available songs (all songs except current user's)
        available_songs = set(str(song_id) for song_id in all_tracks.index) - current_user_songs
        logging.info(f"Available songs after excluding user's history: {len(available_songs)}")

        # Initialize song weights
        song_weights: Dict[str, float] = {str(song_id): 0.0 for song_id in available_songs}
        total_similarity = 0.0
        found_history = False

        # Process each similar user
        for _, user_row in similar_users.iterrows():
            user_id = str(user_row['user_id'])
            similarity = float(user_row['similarity'])
            total_similarity += similarity

            # Try different variants of user ID
            user_variants = [
                user_id,  # Original ID
                f"anon_{user_id}",  # Anonymized version
                user_id.replace('anon_', '')  # De-anonymized version
            ]

            user_history = None
            for variant in user_variants:
                variant_history = listening_history[listening_history['user_id'] == variant]
                if not variant_history.empty:
                    user_history = variant_history
                    logging.info(f"Found listening history for user variant {variant}")
                    break

            if user_history is None or user_history.empty:
                logging.info(f"No listening history found for user {user_id} or its variants")
                continue

            # Process user's history
            found_history = True
            user_songs = set(str(song_id) for song_id in user_history['song_id'])
            for song_id in user_songs:
                if song_id in song_weights:
                    song_weights[song_id] += similarity

        # If no history found for any similar user, assign base scores
        if not found_history:
            logging.warning("No new songs found in similar users' listening history")
            # Assign base scores to all available songs based on average similarity
            avg_similarity = total_similarity / len(similar_users) if len(similar_users) > 0 else 0.3
            base_score = max(0.3, avg_similarity * 0.5)  # At least 0.3, or half of average similarity
            for song_id in available_songs:
                song_weights[song_id] = base_score
        else:
            # Normalize weights by total similarity
            if total_similarity > 0:
                for song_id in song_weights:
                    song_weights[song_id] /= total_similarity

        logging.info(f"Generated scores for {len(song_weights)} unique songs after filtering")
        return song_weights
        
    async def predict(
        self,
        input_data: DemographicInput,
        listening_history: pd.DataFrame,
        all_demographics: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Predict song scores based on demographic similarity.
        Now includes filtering of current user's songs.
        """
        try:
            user_demographics = input_data.demographics
            logging.info(f"Finding similar users for user {input_data.user_id}")
            
            # Find similar users
            similar_users = self._find_similar_users(user_demographics, all_demographics)
            if similar_users.empty:
                logging.warning(f"No similar users found for user {input_data.user_id}")
                return {}
            
            # Analyze listening patterns, excluding current user's songs
            song_scores = self._analyze_listening_patterns(
                similar_users,
                listening_history,
                current_user_id=input_data.user_id
            )
            
            logging.info(f"Generated {len(song_scores)} demographic-based predictions for user {input_data.user_id}")
            return song_scores
            
        except Exception as e:
            logging.error(f"Error in demographic prediction: {str(e)}")
            return {}

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
        
        # Handle empty history case
        if listening_history is None or listening_history.empty:
            logging.warning("Empty listening history provided to popularity predictor")
            return {}
            
        # Get all unique songs
        unique_songs = listening_history['song_id'].unique()
        logging.info(f"Calculating popularity scores for {len(unique_songs)} unique songs")
        
        # Calculate base popularity for each song
        for song_id in unique_songs:
            song_history = listening_history[listening_history['song_id'] == song_id]
            
            # Calculate recency-weighted score
            total_weight = 0
            for _, interaction in song_history.iterrows():
                try:
                    # Convert timestamp to datetime if it's a string
                    if isinstance(interaction['timestamp'], str):
                        interaction_time = pd.to_datetime(interaction['timestamp'])
                    else:
                        interaction_time = interaction['timestamp']
                        
                    age_days = (current_time - interaction_time).days
                    if age_days <= self.max_age_days:
                        weight = self._calculate_time_weight(age_days)
                        song_scores[str(song_id)] += weight
                        total_weight += 1
                        logging.debug(f"Song {song_id} age: {age_days} days, weight: {weight}")
                except (TypeError, AttributeError, ValueError) as e:
                    logging.warning(f"Error processing timestamp for song {song_id}: {str(e)}")
                    song_scores[str(song_id)] += 0.5
                    total_weight += 1
            
            # Normalize by number of interactions
            if total_weight > 0:
                song_scores[str(song_id)] /= total_weight
                logging.debug(f"Final normalized score for song {song_id}: {song_scores[str(song_id)]}")
        
        # If no scores calculated, give equal base scores
        if not song_scores and unique_songs.size > 0:
            logging.info("No scores calculated, using default scores")
            return {str(song_id): 0.5 for song_id in unique_songs}
        
        # Normalize all scores to [0,1] range
        max_score = max(song_scores.values()) if song_scores else 1.0
        if max_score > 0:
            song_scores = {
                song_id: score/max_score 
                for song_id, score in song_scores.items()
            }
            logging.info(f"Normalized popularity scores: {song_scores}")
        
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
            return song_scores #Return the song scores as is if no user context or demographics
            
        demographic_predictor = DemographicPredictor()
        similar_users = demographic_predictor._find_similar_users(
            user_context['demographics'],
            demographics
        )
        
        if similar_users.empty:
            return song_scores #Return the song scores as is if no similar users found
            
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
        logging.info("Starting popularity prediction")
        logging.info(f"Input data: time_window={input_data.time_window}, context={input_data.user_context is not None}")
        
        # Handle bootstrap case
        if listening_history is None or listening_history.empty:
            logging.warning("No listening history available for popularity calculation")
            return {}
            
        # Calculate base popularity scores
        current_time = datetime.now()
        song_scores = self._calculate_popularity_scores(
            listening_history,
            current_time
        )
        
        # If no scores, return default scores for all songs in history
        if not song_scores and not listening_history.empty:
            unique_songs = listening_history['song_id'].unique()
            logging.info("Using default scores for songs in history")
            song_scores = {str(song_id): 0.5 for song_id in unique_songs}
        
        # Apply demographic filtering if context available
        if input_data.user_context and demographics is not None and not demographics.empty:
            logging.info("Applying demographic filtering to popularity scores")
            song_scores = self._apply_demographic_filter(
                song_scores,
                input_data.user_context,
                listening_history,
                demographics
            )
        
        logging.info(f"Final popularity scores: {song_scores}")
        return song_scores 