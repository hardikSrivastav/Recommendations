import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from application.models.fma_dataset_processor import FMADatasetProcessor 
from application.models.model import RecommenderSystem
from application.models.predictors import DemographicPredictor, PopularityPredictor, DemographicInput, PopularityInput
import logging
from sklearn.preprocessing import LabelEncoder
from application.database import DatabaseService
from application.database.postgres_service import PostgresService
from sqlalchemy import text

class IntegratedMusicRecommender:
    def __init__(self, base_dir='./fma_metadata'):
        self.dataset_processor = FMADatasetProcessor(base_dir)
        # Initialize recommender with only embedding and demographic dimensions
        self.recommender = RecommenderSystem(embedding_dim=20)  # Simplified model
        self.demographic_predictor = DemographicPredictor()
        self.popularity_predictor = PopularityPredictor()
        self.pg_service = PostgresService()
        
    def prepare_listening_history(self, tracks_data, num_users=5, interactions_per_user=3):
        """Generate synthetic listening history from FMA data"""
        logging.info("Starting to prepare listening history...")
        
        if len(tracks_data) == 0:
            raise ValueError("No tracks data available")
            
        # Take a small subset of tracks for faster processing
        max_tracks = min(1000, len(tracks_data))
        tracks_subset = tracks_data.head(max_tracks)
        logging.info(f"Using {max_tracks} tracks for initial training")
        
        # Get available song IDs (these are the indices from the original dataset)
        available_song_ids = tracks_subset.index.tolist()
        logging.info(f"Number of available songs: {len(available_song_ids)}")
        
        # Generate synthetic listening history
        listening_history = []
        user_ids = [f"user_{i}" for i in range(num_users)]
        
        for user_id in user_ids:
            # Sample random tracks for each user
            try:
                user_song_ids = np.random.choice(available_song_ids, size=interactions_per_user, replace=False)
                for song_id in user_song_ids:
                    listening_history.append({
                        'user_id': user_id,
                        'song_id': int(song_id),  # Ensure song_id is an integer
                        'timestamp': datetime.now()
                    })
            except ValueError as e:
                logging.error(f"Error sampling tracks for user {user_id}: {str(e)}")
                raise
                
        result_df = pd.DataFrame(listening_history)
        if len(result_df) == 0:
            raise ValueError("Failed to generate listening history")
            
        logging.info(f"Generated listening history with {len(result_df)} entries for {num_users} users")
        logging.info(f"Sample of listening history:\n{result_df.head().to_string()}")
        return result_df
        
    def generate_demographics(self, user_ids):
        """Generate synthetic demographics for users"""
        logging.info("Generating demographics data...")
        
        if not isinstance(user_ids, (list, np.ndarray)) or len(user_ids) == 0:
            raise ValueError("No user IDs provided for demographics generation")
            
        age_ranges = np.array([18, 25, 35, 50, 65])
        genders = ['M', 'F', 'NB', 'O']
        locations = ['US', 'UK', 'CA', 'AU', 'DE']
        occupations = ['Student', 'Professional', 'Artist', 'Engineer', 'Teacher']
        
        demographics = []
        for user_id in user_ids:
            age = int(np.random.choice(age_ranges))
            # Add age group based on age
            age_group = (
                '18-24' if age < 25 else
                '25-34' if age < 35 else
                '35-49' if age < 50 else
                '50-64' if age < 65 else
                '65+'
            )
            
            demographics.append({
                'user_id': str(user_id),  # Ensure user_id is string
                'age': age,  # Ensure age is int
                'age_group': age_group,  # Add age group
                'gender': str(np.random.choice(genders)),  # Ensure gender is string
                'location': str(np.random.choice(locations)),  # Ensure location is string
                'occupation': str(np.random.choice(occupations))  # Ensure occupation is string
            })
            
        result_df = pd.DataFrame(demographics)
        if len(result_df) == 0:
            raise ValueError("Failed to generate demographics data")
            
        # Encode categorical features
        categorical_features = ['age_group', 'gender', 'location', 'occupation']
        
        for feature in categorical_features:
            encoder = LabelEncoder()
            result_df[f'{feature}_encoded'] = encoder.fit_transform(result_df[feature])
            
        logging.info(f"Generated demographics for {len(result_df)} users")
        return result_df
        
    def train_model(self, num_users=5, interactions_per_user=3, epochs=3):
        """Train the recommendation model with minimal data for initial setup"""
        try:
            logging.info("Starting model training process...")
            
            # Load track IDs from PostgreSQL
            tracks_data = self.pg_service.get_tracks()
            if tracks_data.empty:
                # If PostgreSQL is empty, load from file and store
                tracks_data = self.dataset_processor.process_dataset()
                
            if tracks_data.empty:
                raise ValueError("Failed to load tracks data")
                
            logging.info(f"Loaded {len(tracks_data)} tracks")
            
            # Generate synthetic listening history
            listening_history = self.prepare_listening_history(
                tracks_data,
                num_users=num_users,
                interactions_per_user=interactions_per_user
            )
            
            if len(listening_history) == 0:
                raise ValueError("Failed to generate listening history")
                
            # Generate demographics
            unique_users = listening_history['user_id'].unique()
            self.demographics_df = self.generate_demographics(unique_users)  # Store demographics
            
            # Store tracks data for later use
            self.tracks_data = tracks_data
            
            # Train the model
            logging.info("Starting model training...")
            self.recommender.fit(
                listening_history,
                self.demographics_df,
                epochs=epochs,
                batch_size=16  # Small batch size for initial testing
            )
            logging.info("Model training completed successfully")
            
            return self.recommender
            
        except Exception as e:
            logging.error(f"Error in train_model: {str(e)}")
            raise
            
    async def get_recommendations(self, user_id, n=5):
        """Get recommendations for a user"""
        if not hasattr(self, 'recommender') or self.recommender.model is None:
            raise ValueError("Model needs to be trained first")
            
        try:
            # Check if user exists in training data
            if user_id not in self.demographics_df['user_id'].values:
                logging.info(f"New user {user_id} detected. Adding to training data...")
                
                # Generate demographics for new user
                new_user_demographics = self.generate_demographics([user_id])
                self.demographics_df = pd.concat([self.demographics_df, new_user_demographics])
                
                # Generate some random initial listening history for the user
                tracks_data = self.pg_service.get_tracks()
                new_user_history = self.prepare_listening_history(
                    tracks_data,
                    num_users=1,
                    interactions_per_user=5
                )
                new_user_history['user_id'] = user_id  # Override generated user_id
                
                # Retrain model with new user data
                logging.info("Retraining model with new user data...")
                self.recommender.fit(
                    new_user_history,
                    self.demographics_df,
                    epochs=1,  # Quick update for new user
                    batch_size=16
                )
            
            # Now get recommendations
            return await self.recommender.predict_next_songs(
                user_id,
                self.demographics_df,
                n=n
            )
        except Exception as e:
            logging.error(f"Error getting recommendations: {str(e)}")
            raise

    async def get_ensemble_recommendations(self, user_id, n=5):
        """Get recommendations using all predictors with weights and confidence scores"""
        try:
            # Get current timestamp
            current_time = datetime.now()
            
            # Get user's actual listening history from database
            db_service = DatabaseService()
            actual_history = db_service.get_listening_history(str(user_id))
            has_listening_history = bool(actual_history and len(actual_history) > 0)
            
            # Convert MongoDB history to DataFrame
            listening_history = pd.DataFrame(actual_history) if actual_history else pd.DataFrame()
            
            # Get user's demographics from PostgreSQL
            user_demographics = db_service.get_demographics(str(user_id))
            has_demographics = bool(user_demographics and len(user_demographics) > 0)
            
            # Get all users' demographics from PostgreSQL
            session = db_service.Session()
            try:
                # Query all demographics
                query = text("""
                    SELECT 
                        user_id,
                        age,
                        gender,
                        location,
                        occupation,
                        created_at,
                        updated_at
                    FROM user_demographics
                """)
                result = session.execute(query)
                
                # Convert to list of dicts
                all_users_demographics = [
                    {
                        'user_id': row.user_id,
                        'age': row.age,
                        'gender': row.gender,
                        'location': row.location,
                        'occupation': row.occupation
                    }
                    for row in result
                ]
                
                demographics_df = pd.DataFrame(all_users_demographics) if all_users_demographics else None
                logging.info(f"Loaded {len(all_users_demographics)} user demographics from PostgreSQL")
            finally:
                session.close()
            
            # Determine cold start based on both history and demographics
            is_cold_start = not (has_listening_history and has_demographics)
            
            if is_cold_start:
                logging.info(f"Cold start detected for user {user_id}. History: {has_listening_history}, Demographics: {has_demographics}")
                
                # For cold starts, use existing demographics if available, otherwise use defaults
                if not has_demographics:
                    user_demographics = {
                        'user_id': str(user_id),
                        'age': 25,  # Default age
                        'age_group': '25-34',  # Default age group
                        'gender': 'unknown',
                        'location': 'unknown',
                        'occupation': 'unknown'
                    }
                    
                    # Add to demographics_df if it exists
                    if demographics_df is not None:
                        demographics_df = pd.concat([
                            demographics_df,
                            pd.DataFrame([user_demographics])
                        ], ignore_index=True)
                    else:
                        demographics_df = pd.DataFrame([user_demographics])
                
                # Set predictor weights for cold start
                predictor_weights = {
                    'model': 0.2,      # Lower weight for model in cold start
                    'demographic': 0.3, # Medium weight for demographics
                    'popularity': 0.5   # Higher weight for popularity
                }
            else:
                # Normal weights for existing users
                predictor_weights = {
                    'model': 0.5,      # Higher weight for model
                    'demographic': 0.3, # Medium weight for demographics
                    'popularity': 0.2   # Lower weight for popularity
                }
            
            # Get recent songs from history
            recent_songs = []
            if actual_history:
                recent_songs = [str(entry['song_id']) for entry in actual_history][:10]  # Last 10 songs
            
            # Get predictions from each source
            predictions = []
            
            # 1. Model predictions (minimal weight for cold start)
            model_scores = {}
            if hasattr(self, 'recommender') and self.recommender.model is not None:
                try:
                    # Check if user_id exists in model's encoder
                    if user_id not in self.recommender.user_encoder.classes_:
                        # Add the new user to the encoder
                        new_classes = np.append(self.recommender.user_encoder.classes_, [user_id])
                        self.recommender.user_encoder.fit(new_classes)
                    
                    # Use demographics directly without encoding
                    model_preds = await self.recommender.predict_next_songs(
                        user_id,
                        user_demographics,
                        n=n * 2  # Get more predictions to allow for blending
                    )
                    for pred in model_preds:
                        model_scores[pred['song_id']] = pred['score']
                except Exception as e:
                    logging.warning(f"Model predictions failed, falling back to other predictors: {str(e)}")
            
            # 2. Demographic predictions
            demographic_scores = {}
            if demographics_df is not None:
                try:
                    demographic_input = DemographicInput(
                        user_id=str(user_id),
                        demographics=user_demographics,
                        timestamp=current_time
                    )
                    demographic_scores = await self.demographic_predictor.predict(
                        demographic_input,
                        listening_history,
                        demographics_df
                    )
                except Exception as e:
                    logging.warning(f"Demographic predictions failed: {str(e)}")
                    logging.warning(f"Demographics data: user={user_demographics}, all={demographics_df.head()}")
            
            # 3. Popularity predictions
            popularity_scores = {}
            try:
                popularity_input = PopularityInput(
                    time_window=timedelta(days=30),
                    user_context={'demographics': user_demographics}
                )
                popularity_scores = await self.popularity_predictor.predict(
                    popularity_input,
                    listening_history,
                    demographics_df
                )
            except Exception as e:
                logging.warning(f"Popularity predictions failed: {str(e)}")
            
            # Combine all unique song IDs
            all_songs = set(model_scores.keys()) | set(demographic_scores.keys()) | set(popularity_scores.keys())
            
            if not all_songs:
                # Fallback to random sampling from available songs
                logging.warning("No predictions available from any source, falling back to random sampling")
                tracks = self.pg_service.get_tracks()
                available_songs = tracks.index.tolist()
                fallback_songs = np.random.choice(available_songs, size=min(1000, len(available_songs)), replace=False)
                all_songs = set(str(song_id) for song_id in fallback_songs)
                
                # Generate random scores for fallback
                for song_id in all_songs:
                    model_scores[song_id] = np.random.beta(2, 5)
                    demographic_scores[song_id] = np.random.beta(2, 5)
                    popularity_scores[song_id] = np.random.beta(2, 5)
            
            # Calculate combined scores with individual predictor contributions
            combined_predictions = []
            for song_id in all_songs:
                # Get individual scores with defaults
                model_score = model_scores.get(str(song_id), 0.0)
                demographic_score = demographic_scores.get(str(song_id), 0.0)
                popularity_score = popularity_scores.get(str(song_id), 0.0)
                
                # Calculate weighted score
                weighted_score = (
                    model_score * predictor_weights['model'] +
                    demographic_score * predictor_weights['demographic'] +
                    popularity_score * predictor_weights['popularity']
                )
                
                # Store individual predictor contributions
                predictor_contributions = {
                    'model': model_score,
                    'demographic': demographic_score,
                    'popularity': popularity_score
                }
                
                combined_predictions.append({
                    'song_id': str(song_id),
                    'confidence': float(weighted_score),
                    'predictor_weights': predictor_weights,
                    'predictor_scores': predictor_contributions,
                    'was_shown': False,
                    'was_selected': False
                })
            
            # Sort by confidence and get top N
            combined_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            top_predictions = combined_predictions[:n]
            
            # Create final response format
            response = {
                'user_id': str(user_id),
                'timestamp': current_time.isoformat(),
                'context': {
                    'total_songs': len(listening_history),
                    'recent_songs': recent_songs,
                    'demographics': user_demographics,
                    'is_cold_start': is_cold_start
                },
                'predictions': top_predictions
            }
            
            return response
            
        except Exception as e:
            logging.error(f"Error generating ensemble predictions: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    async def main():
        integrated_recommender = IntegratedMusicRecommender()
        model = integrated_recommender.train_model(num_users=10, interactions_per_user=15, epochs=10)
        dataset_processor = FMADatasetProcessor(base_dir='./fma_metadata')
        recommendations = await integrated_recommender.get_recommendations("user_0", n=5)
        print(f"Recommendations for user_0: {recommendations}")
    
    import asyncio
    asyncio.run(main())
    