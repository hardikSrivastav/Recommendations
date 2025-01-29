import pandas as pd
import numpy as np
import torch
from datetime import datetime
from application.models.fma_dataset_processor import FMADatasetProcessor 
from application.models.model import RecommenderSystem
import logging

class IntegratedMusicRecommender:
    def __init__(self, base_dir='./fma_metadata'):
        self.dataset_processor = FMADatasetProcessor(base_dir)
        self.recommender = RecommenderSystem(embedding_dim=50, metadata_dim=2, demographic_dim=4)
        
    def prepare_listening_history(self, tracks_data, num_users=1000, interactions_per_user=10):
        """Generate synthetic listening history from FMA data"""
        user_ids = [f"user_{i}" for i in range(num_users)]
        listening_history = []
        
        for user_id in user_ids:
            # Sample random tracks for each user
            user_tracks = tracks_data.sample(n=interactions_per_user)
            
            for _, track in user_tracks.iterrows():
                listening_history.append({
                    'user_id': user_id,
                    'song_id': track.name,  # Using track index as song_id
                    'timestamp': datetime.now()  # You could randomize this if needed
                })
                
        return pd.DataFrame(listening_history)

    def prepare_metadata(self, tracks_data):
        """Process FMA metadata into format needed by recommender"""
        # Create metadata DataFrame with track index as song_id
        metadata = pd.DataFrame()
        metadata['song_id'] = tracks_data.index
        metadata['tags'] = tracks_data['track_tags'].fillna('')
        metadata['genres'] = tracks_data['track_genres'].fillna('')
        
        return metadata
    
    def generate_demographics(self, user_ids):
        age_ranges = np.array([18, 25, 35, 50, 65])
        genders = ['M', 'F', 'NB', 'O']
        locations = ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'BR', 'IN']
        occupations = ['Student', 'Professional', 'Artist', 'Engineer', 'Teacher', 'Healthcare', 'Business', 'Service', 'Retired']

        demographics = []
        for user_id in user_ids:
            demographics.append({
                'user_id': user_id,
                'age': np.random.choice(age_ranges),
                'gender': np.random.choice(genders),
                'location': np.random.choice(locations),
                'occupation': np.random.choice(occupations)
            })

        return pd.DataFrame(demographics)

    def train_model(self, num_users=1000, interactions_per_user=10, epochs=10):
        """Train the recommendation model using FMA dataset"""
        # Load and process FMA data
        tracks_data = self.dataset_processor.process_dataset()
        
        # Prepare training data
        listening_history = self.prepare_listening_history(tracks_data, num_users, interactions_per_user)
        self.demographics_df = self.generate_demographics(listening_history['user_id'].unique())
        metadata = self.prepare_metadata(tracks_data)
        logging.debug(f"We're inside the train_model function in the Integrated Music Recommender.\n This is what our listening history is\n: {listening_history} and this is our metadata\n:{metadata}")
        
        # Train the model
        self.recommender.fit(listening_history, metadata, self.demographics_df, epochs=epochs)
        logging.debug(f"Model has been fit. This is what the recommender looks like: {self.recommender}")
        
        return self.recommender

    def get_recommendations(self, user_id, track_data, n=5):
        if not hasattr(self, 'recommender') or self.recommender.model is None:
            raise ValueError("Model needs to be trained first")
        return self.recommender.predict_next_songs(user_id, track_data, self.demographics_df, n=n)  

# Example usage
if __name__ == "__main__":
    integrated_recommender = IntegratedMusicRecommender()
    model = integrated_recommender.train_model(num_users=10, interactions_per_user=15, epochs=10)
    dataset_processor = FMADatasetProcessor(base_dir='./fma_metadata')
    recommendations = integrated_recommender.get_recommendations("user_0", dataset_processor.process_dataset(), n=5)
    print(f"Recommendations for user_0: {recommendations}")
    