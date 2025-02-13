import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import logging
from typing import Dict, List, Any, Optional
import ast
from collections import defaultdict
from datetime import datetime
from application.models.fma_dataset_processor import FMADatasetProcessor
from application.models.predictors import PopularityPredictor
from application.models.predictors import DemographicPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ConfidenceCalculator:
    """Calculates confidence scores for predictions based on multiple factors."""
    
    def __init__(self):
        self.history_weight = 0.4  # Weight for user history length
        self.embedding_weight = 0.3  # Weight for embedding similarity
        self.diversity_weight = 0.3  # Weight for prediction diversity
        
    def calculate_confidence(
        self,
        predictions: Dict[str, float],
        pred_type: str,
        user_context: Dict[str, Any]
    ) -> float:
        """
        Calculate confidence score for a set of predictions.
        
        Mathematical formulation:
        confidence = w_h * f_h + w_e * f_e + w_d * f_d
        where:
        - w_h, w_e, w_d are weights for history, embedding, and diversity
        - f_h = min(1, log(1 + history_length) / log(50))  <-  Normalized history factor
        - f_e = cosine_similarity(user_embedding, song_embedding) 
        - f_d = 1 - (std(prediction_scores) / max_possible_std)  # Diversity factor
        """
        history_factor = self._calculate_history_factor(user_context.get('history_length', 0))
        embedding_factor = self._calculate_embedding_factor(predictions, user_context)
        diversity_factor = self._calculate_diversity_factor(predictions)
        
        confidence = (
            self.history_weight * history_factor +
            self.embedding_weight * embedding_factor +
            self.diversity_weight * diversity_factor
        )
        
        return min(1.0, max(0.0, confidence)) # 0 < confidence < 1
    
    def _calculate_history_factor(self, history_length: int) -> float:
        """
        Calculate confidence factor based on user history length.
        Uses logarithmic scaling to handle varying history lengths.
        """

        """
        log1p(x) = log(1+x), which means that when there are no songs added, log1p(0) = 0, rather than log(0) = -infinity, which'll throw an error.
        divide by log(50) to normalize the history length, because the max history length is 50 songs.
        """

        return min(1.0, np.log1p(history_length) / np.log(50)) # 0< hf < 1
    
    def _calculate_embedding_factor(
        self,
        predictions: Dict[str, float],
        user_context: Dict[str, Any]
    ) -> float:
        """
        Calculate confidence factor based on embedding similarity.
        Uses normalized cosine similarity between user and song embeddings.
        
        Mathematical formulation:
        1. Normalize embeddings: e_norm = e /||e|| -> unit vector embeddings
        2. Calculate cosine similarity: cos_sim = (u_norm · s_norm) -> dot product of the unit vector embeddings
        3. Scale to [0,1]: similarity = (cos_sim + 1) / 2 -> scale to 0-1
        where:
        - e is the embedding vector
        - ||e|| is the L2 norm of the vector
        - u_norm is the normalized user embedding
        - s_norm is the normalized song embedding
        """
        if 'user_embedding' not in user_context:
            return 0.5  # Default if no embedding available
            
        similarities = []
        for song_id in predictions:
            if 'song_embeddings' in user_context and song_id in user_context['song_embeddings']:
                user_emb = user_context['user_embedding']
                song_emb = user_context['song_embeddings'][song_id]
                
                # Normalize embeddings
                user_emb_norm = user_emb / torch.norm(user_emb)
                song_emb_norm = song_emb / torch.norm(song_emb)
                
                # Calculate cosine similarity and scale to [0,1]
                similarity = torch.cosine_similarity(
                    user_emb_norm.unsqueeze(0),
                    song_emb_norm.unsqueeze(0)
                ).item()
                
                # Scale from [-1,1] to [0,1]
                scaled_similarity = (similarity + 1) / 2
                similarities.append(scaled_similarity)
                
        return np.mean(similarities) if similarities else 0.5
    
    def _calculate_diversity_factor(self, predictions: Dict[str, float]) -> float:
        """
        Calculate confidence factor based on prediction diversity.
        Uses standard deviation of prediction scores normalized by maximum possible std.
        """
        scores = list(predictions.values())
        if not scores:
            return 0.5
            
        std = np.std(scores)
        max_possible_std = 0.5  # Maximum possible std for scores in [0,1]
        return 1 - (std / max_possible_std)  # Higher diversity -> lower confidence

class WeightedEnsembleRecommender:
    """
    Handles dynamic weighting of different recommendation sources based on 
    confidence scores and user context.
    
    Mathematical formulation for final scores:
    score(song_i) = Σ(w_j * s_ij) for j in prediction_sources
    where:
    - w_j is the weight for prediction source j
    - s_ij is the score for song i from source j
    - weights w_j are normalized confidence scores: w_j = conf_j / Σ(conf_k)
    """
    
    def __init__(self):
        self.confidence_calculator = ConfidenceCalculator()
        self.prediction_sources = {
            'model': None,  # Will be set to MusicRecommender instance
            'demographic': None, # Will be set to DemographicPredictor instance
            'popularity': None  #Will be set to PopularityPredictor instance
        }
        
    def set_model_predictor(self, model_predictor: 'MusicRecommender'):
        self.prediction_sources['model'] = model_predictor

    def set_demographic_predictor(self, demographic_predictor: 'DemographicPredictor'):
        self.prediction_sources['demographic'] = demographic_predictor

    def set_popularity_predictor(self, popularity_predictor: 'PopularityPredictor'):
        self.prediction_sources['popularity'] = popularity_predictor
        
    async def get_recommendations(
        self,
        user_id: str,
        user_context: Dict[str, Any],
        n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get weighted recommendations from all available sources.
        
        Process:
        1. Get predictions from each source
        2. Calculate confidence scores
        3. Blend predictions using dynamic weights
        4. Return top N recommendations with metadata
        """
        predictions = {}
        confidence_scores = {}
        
        # Get predictions from each source
        for source_name, predictor in self.prediction_sources.items():
            # source_names -> Model, Demographic, Popularity
            if predictor is not None:
                try:
                    preds = await self._get_source_predictions(predictor, user_id, user_context)
                    confidence = self.confidence_calculator.calculate_confidence(
                        preds,
                        source_name,
                        user_context
                    )
                    predictions[source_name] = preds
                    confidence_scores[source_name] = confidence
                except Exception as e:
                    logging.error(f"Error getting predictions from {source_name}: {str(e)}")
                    
        # Blend predictions
        blended_scores = self._blend_predictions(predictions, confidence_scores)
        
        # Get top N recommendations
        top_n = sorted(
            blended_scores.items(), # For each song, get the song_id and the data
            key=lambda x: x[1]['score'], # Rank by score
            reverse=True # highest -> lowest
        )[:n]
        
        return [{
            'song_id': song_id,
            'score': data['score'],
            'confidence': data['confidence'],
            'source_weights': data['source_weights']
        } for song_id, data in top_n]
    
    async def _get_source_predictions(
        self,
        predictor: Any,
        user_id: str,
        user_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Get predictions from a single source."""
        if hasattr(predictor, 'predict_async'): # if the predictor operates asynchronously
            return await predictor.predict_async(user_id, user_context)
        return predictor.predict(user_id, user_context)
    
    def _blend_predictions(
        self,
        predictions: Dict[str, Dict[str, float]],
        confidence_scores: Dict[str, float]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Blend predictions from multiple sources using confidence scores as weights.
        
        1. Normalize confidence scores to get weights:
           w_j = conf_j / Σ(conf_k)
        2. For each song, calculate weighted score:
           score_i = Σ(w_j * s_ij)
        3. Calculate final confidence as weighted average of source confidences
        w_j -> weights of predictions from j sources
        s_ij -> scores for i songs from j sources
        score_i -> final combined scores for i songs
        """
        # Calculate normalized weights
        total_confidence = sum(confidence_scores.values())
        weights = {
            source: score / total_confidence
            for source, score in confidence_scores.items()
        }
        
        # Blend scores
        blended_scores = defaultdict(lambda: {
            'score': 0.0,
            'confidence': 0.0,
            'source_weights': {}
        })
        
        # Calculate weighted scores and track source contributions
        for source, preds in predictions.items():
            source_weight = weights[source]
            for song_id, score in preds.items():
                blended_scores[song_id]['score'] += score * source_weight
                blended_scores[song_id]['confidence'] += confidence_scores[source] * source_weight
                blended_scores[song_id]['source_weights'][source] = source_weight
                
        return blended_scores

class MusicRecommender(nn.Module):
    def __init__(self, num_users, num_songs, embedding_dim, demographic_dim, metadata_dim=64):
        super(MusicRecommender, self).__init__()
        
        # Log initial dimensions
        logging.info(f"Initializing MusicRecommender with:")
        logging.info(f"num_users: {num_users}")
        logging.info(f"num_songs: {num_songs}")
        logging.info(f"embedding_dim: {embedding_dim}")
        logging.info(f"demographic_dim: {demographic_dim}")
        logging.info(f"metadata_dim: {metadata_dim}")
        
        # Basic embeddings for users and songs
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.song_embedding = nn.Embedding(num_songs, embedding_dim)
        
        # Process metadata first to reduce dimensionality

        self.metadata_reducer = nn.Sequential(
            nn.Linear(metadata_dim, embedding_dim), # make the metadata dimension the same as the embedding dimension
            nn.ReLU(), # introduce non-linearity
            nn.Dropout(0.2) # prevent overfitting
        )
        
        # Calculate total input dimension
        total_input_dim = embedding_dim * 3 + demographic_dim  # (user_emb + song_emb + metadata_emb) = emb_dim + demographics
        
        logging.info(f"Total input dimension: {total_input_dim}")
        
        # Network architecture
        self.network = nn.Sequential(
            # First layer: reduce to 128
            nn.Linear(total_input_dim, 128),
            nn.LayerNorm(128), # normalize the input
            nn.ReLU(),
            nn.Dropout(0.2),

            # Second layer: reduce to 64
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Third layer: reduce to 32
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output layer
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, user_input, song_input, metadata_input, demographic_input):
        """Forward pass with metadata features"""
        # Log input shapes
        logging.debug(f"Input shapes:")
        logging.debug(f"user_input: {user_input.shape}")
        logging.debug(f"song_input: {song_input.shape}")
        logging.debug(f"metadata_input: {metadata_input.shape}")
        logging.debug(f"demographic_input: {demographic_input.shape}")
        
        # Get embeddings
        user_embedded = self.user_embedding(user_input)  # [batch_size, embedding_dim]
        song_embedded = self.song_embedding(song_input)  # [batch_size, embedding_dim]
        
        # Process metadata
        metadata_embedded = self.metadata_reducer(metadata_input)  # [batch_size, embedding_dim]
        
        # Handle NaN values in demographics
        demographic_input = torch.nan_to_num(demographic_input, 0.0)
        
        # Get batch size from the first input
        batch_size = user_input.size(0)
        
        # Reshape all tensors to ensure they have the correct batch dimension
        if user_embedded.dim() == 1:
            user_embedded = user_embedded.unsqueeze(0)
        if song_embedded.dim() == 1:
            song_embedded = song_embedded.unsqueeze(0)
        if metadata_embedded.dim() == 1:
            metadata_embedded = metadata_embedded.unsqueeze(0)
        if demographic_input.dim() == 1:
            demographic_input = demographic_input.unsqueeze(0)
            
        # Ensure all tensors have the correct batch size
        if user_embedded.size(0) != batch_size:
            user_embedded = user_embedded.expand(batch_size, -1)
        if song_embedded.size(0) != batch_size:
            song_embedded = song_embedded.expand(batch_size, -1)
        if metadata_embedded.size(0) != batch_size:
            metadata_embedded = metadata_embedded.expand(batch_size, -1)
        if demographic_input.size(0) != batch_size:
            demographic_input = demographic_input.expand(batch_size, -1)
            
        # Log shapes after processing
        logging.debug(f"Processed shapes:")
        logging.debug(f"user_embedded: {user_embedded.shape}")
        logging.debug(f"song_embedded: {song_embedded.shape}")
        logging.debug(f"metadata_embedded: {metadata_embedded.shape}")
        logging.debug(f"demographic_input: {demographic_input.shape}")
        
        # Concatenate all inputs along feature dimension
        combined = torch.cat([
            user_embedded, # (emb_dim, batch_size)
            song_embedded, # (emb_dim, batch_size)
            metadata_embedded, # (emb_dim, batch_size)
            demographic_input # (demographic_dim, batch_size)
        ], dim=1  # concatenate along batch_size dimension
        )
        
        logging.debug(f"Combined shape: {combined.shape}")
        
        # Forward through network
        output = self.network(combined)
        
        # Ensure output has correct shape [batch_size, 1]
        if output.dim() == 1:
            output = output.unsqueeze(1)
            
        logging.debug(f"Final output shape: {output.shape}")
        
        return output

class RecommenderSystem:
    def __init__(self, embedding_dim=20):
        self.embedding_dim = embedding_dim
        self.demographic_dim = 4  # age_group, gender, location, occupation
        self.user_encoder = LabelEncoder()
        self.song_encoder = LabelEncoder()
        self.encoders = {
            'age_group': LabelEncoder(),
            'gender': LabelEncoder(),
            'location': LabelEncoder(),
            'occupation': LabelEncoder()
        }
        self.model = None
        self.fma_processor = FMADatasetProcessor()
        self.metadata_features = None
        
    def encode_demographics(self, demographics_df):
        """Encode demographic features"""
        encoded_df = demographics_df.copy()
        
        # Ensure all necessary columns exist first
        if 'gender' not in encoded_df:
            encoded_df['gender'] = 'unknown'
        if 'location' not in encoded_df:
            encoded_df['location'] = 'unknown'
        if 'occupation' not in encoded_df:
            encoded_df['occupation'] = 'unknown'
        
        # Create age groups with a list instead of categorical
        age_bins = [0, 18, 25, 35, 50, 100]
        age_labels = ['<18', '18-25', '26-35', '36-50', '50+']
        encoded_df['age_group'] = 'unknown'  # Default value
        
        # Assign age groups manually to avoid categorical issues
        for i in range(len(age_bins)-1):
            mask = (encoded_df['age'] >= age_bins[i]) & (encoded_df['age'] < age_bins[i+1])
            encoded_df.loc[mask, 'age_group'] = age_labels[i]
        
        # Initialize encoders if not already done
        if not hasattr(self, 'encoders'):
            self.encoders = {
                'age_group': LabelEncoder(),
                'gender': LabelEncoder(),
                'location': LabelEncoder(),
                'occupation': LabelEncoder()
            }
        
        # Encode each demographic feature
        for col, encoder in self.encoders.items():
            # Get the feature name without '_encoded' suffix
            feature = col.replace('_encoded', '')
            
            # Fill NaN values with 'unknown'
            encoded_df[feature] = encoded_df[feature].fillna('unknown')
            
            # Fit the encoder if it hasn't been fitted yet
            if not hasattr(encoder, 'classes_'):
                # Add unknown class to ensure it's always available
                unique_values = np.unique(encoded_df[feature].values)
                if 'unknown' not in unique_values:
                    unique_values = np.append(unique_values, 'unknown')
                encoder.fit(unique_values)
            
            # Transform the values
            try:
                encoded_df[f'{col}_encoded'] = encoder.transform(encoded_df[feature])
            except ValueError:
                # If we encounter unknown values, add them to the encoder
                unique_values = np.unique(encoded_df[feature].values)
                encoder.fit(np.append(encoder.classes_, unique_values))
                encoded_df[f'{col}_encoded'] = encoder.transform(encoded_df[feature])
        
        return encoded_df

    def generate_training_data(self, listening_history, demographics_df):
        """Generate training data with metadata features"""
        try:
            users, songs, metadata, demographics, labels = [], [], [], [], []
            
            # Merge demographics with listening history
            listening_history = listening_history.merge(
                demographics_df,
                on='user_id',
                how='left'
            )
            
            logging.info(f"Merged data shape: {listening_history.shape}")
            
            # Process each user's data
            for user in listening_history['user_encoded'].unique():
                user_history = listening_history[listening_history['user_encoded'] == user]
                if user_history.empty:
                    continue
                    
                user_demographics = user_history[['age_group_encoded', 'gender_encoded', 'location_encoded', 'occupation_encoded']].iloc[0]
                
                # Add positive samples
                for _, row in user_history.iterrows():
                    song_id = int(row['song_id'])
                    song_metadata = self.get_song_metadata(song_id)
                    
                    if song_metadata is not None:
                        # Convert metadata to feature vector
                        metadata_vector = self._metadata_to_vector(song_metadata)
                        
                        users.append(user)
                        songs.append(row['song_encoded'])
                        metadata.append(metadata_vector)
                        demographics.append(user_demographics.values)
                        labels.append(1)
                        
                        # Add negative samples
                        user_songs = set(user_history['song_encoded'].values)
                        all_songs = set(range(max(listening_history['song_encoded']) + 1))
                        available_songs = list(all_songs - user_songs)

                        """
                        Negative samples are anti-theses to positive samples: our user(s) never actually listened to these songs, we're adding them to show the Neural Network that, presumably, "this is what the user didn't like."
                        There is one 'negative interaction' for every positive interaction. So, each user has 2x numbers of interactions, if the interactions per user parameter = x. Negative sample songs are taken from the complementary set of songs in listening history
                            set(ns[songs]) = U - {set(listening_history[songs])}    
                                ^negative samples      ^positive samples
                        The labels array is a boolean array (has either 0, 1). 0 -> Negative Sample; 1 -> Positive Sample
                        """
                        
                        if available_songs:
                            negative_song = int(np.random.choice(available_songs))
                            negative_song_id = int(self.song_encoder.inverse_transform([negative_song])[0]) # Reverse what we did while transforming the song_id on line ~690
                            negative_metadata = self.get_song_metadata(negative_song_id)
                            
                            if negative_metadata is not None:
                                negative_metadata_vector = self._metadata_to_vector(negative_metadata)
                                
                                users.append(user)
                                songs.append(negative_song)
                                metadata.append(negative_metadata_vector)
                                demographics.append(user_demographics.values)
                                labels.append(0)
            
            if not users:
                raise ValueError("No training data generated")
                
            logging.info(f"Generated {len(users)} training samples")
            
            # Convert lists to numpy arrays first
            users_array = np.array(users, dtype=np.int64)
            songs_array = np.array(songs, dtype=np.int64)
            metadata_array = np.stack(metadata)  # Stack metadata vectors into a single array
            demographics_array = np.stack(demographics)  # Stack demographics into a single array
            labels_array = np.array(labels, dtype=np.float32)
            
            # Ensure all arrays have the same first dimension
            n_samples = len(users_array)
            assert len(songs_array) == n_samples
            assert len(metadata_array) == n_samples
            assert len(demographics_array) == n_samples
            assert len(labels_array) == n_samples
            
            logging.info(f"Array shapes - Users: {users_array.shape}, Songs: {songs_array.shape}, "
                        f"Metadata: {metadata_array.shape}, Demographics: {demographics_array.shape}, "
                        f"Labels: {labels_array.shape}")
            
            # Convert numpy arrays to tensors with explicit shapes
            return (
                torch.tensor(users_array, dtype=torch.long),                    # [batch_size]
                torch.tensor(songs_array, dtype=torch.long),                    # [batch_size]
                torch.tensor(metadata_array, dtype=torch.float32),              # [batch_size, metadata_dim]
                torch.tensor(demographics_array, dtype=torch.float32),          # [batch_size, demographic_dim]
                torch.tensor(labels_array, dtype=torch.float32).unsqueeze(1)    # [batch_size, 1]
            )
            
        except Exception as e:
            logging.error(f"Error generating training data: {str(e)}")
            raise
            
    def _metadata_to_vector(self, metadata):
        """Convert song metadata to a fixed-size feature vector"""
        # Initialize a zero vector
        vector = np.zeros(64)  # Fixed size for metadata features
        
        try:
            # Process genres (first 20 dimensions)
            genres = metadata['genres']
            if isinstance(genres, list) and len(genres) > 0:
                for i, genre in enumerate(genres[:10]):  # Use first 10 genres
                    # Use a deterministic hash with a fixed seed
                    genre_hash = hash(str(genre) + "genre_seed_42") % 2  # Fixed seed for reproducibility
                    vector[i] = genre_hash
                    
            # Process tags (next 20 dimensions)
            tags = metadata['tags']
            if isinstance(tags, list) and len(tags) > 0:
                for i, tag in enumerate(tags[:10]):  # Use first 10 tags
                    vector[20 + i] = hash(str(tag)) % 2  # Convert tag to binary feature
                    
            # Add duration (normalized) (1 dimension)
            duration = float(metadata['duration'])
            vector[40] = min(duration / 3600, 1.0)  # Normalize to [0,1], cap at 1 hour
            
            # One-hot encode top genre (remaining dimensions)
            genre_top = str(metadata['genre_top'])
            genre_hash = hash(f'{genre_top} genre_seed_42') % 23  # Use 23 dimensions for top genre, 0 < genre_hash < 22
            vector[41 + genre_hash] = 1 # Set the corresponding dimension to 1
            
        except Exception as e:
            logging.warning(f"Error converting metadata to vector: {str(e)}")
            # Return zero vector in case of error
            
        return vector

    def fit(self, listening_history, demographics_df, epochs=5, batch_size=32):
        """Train the model with metadata features"""
        try:
            # Process metadata first
            self.process_metadata()
            if self.metadata_features is None:
                raise ValueError("Failed to process metadata features")
            
            logging.info(f"Metadata features shape: {self.metadata_features.shape}")
            logging.info(f"Metadata features columns: {self.metadata_features.columns.tolist()}")
            
            # Encode user IDs
            self.user_encoder.fit(listening_history['user_id'].astype(str))
            listening_history['user_encoded'] = self.user_encoder.transform(listening_history['user_id'].astype(str))
            
            # Encode song IDs
            self.song_encoder.fit(listening_history['song_id'].astype(int))
            listening_history['song_encoded'] = self.song_encoder.transform(listening_history['song_id'].astype(int))
            
            logging.info(f"Listening history shape: {listening_history.shape}")
            logging.info(f"Listening history columns: {listening_history.columns.tolist()}")
            logging.info(f"Sample listening history:\n{listening_history.head().to_dict()}")
            
            # Process demographics
            processed_demographics = self.encode_demographics(demographics_df)
            logging.info(f"Processed demographics shape: {processed_demographics.shape}")
            logging.info(f"Processed demographics columns: {processed_demographics.columns.tolist()}")
            
            num_users = len(self.user_encoder.classes_)
            num_songs = len(self.song_encoder.classes_)
            metadata_dim = 64  # Fixed dimension for metadata features
            
            logging.info(f"Training model with {num_users} users and {num_songs} songs")
            logging.info(f"Using batch size: {batch_size}, epochs: {epochs}")
            logging.info(f"Metadata dimension: {metadata_dim}")
            logging.info(f"Embedding dimension: {self.embedding_dim}")
            logging.info(f"Demographic dimension: {self.demographic_dim}")
            
            # Initialize model with metadata dimension
            self.model = MusicRecommender(
                num_users,
                num_songs,
                self.embedding_dim,
                self.demographic_dim,
                metadata_dim
            )
            
            optimizer = optim.Adam(self.model.parameters())
            criterion = nn.BCELoss()
            
            # Generate training data with metadata
            users, songs, metadata, demographics, labels = self.generate_training_data(
                listening_history,
                processed_demographics
            )
            
            # Log tensor shapes
            logging.info("Training data tensor shapes:")
            logging.info(f"Users tensor: {users.shape}")
            logging.info(f"Songs tensor: {songs.shape}")
            logging.info(f"Metadata tensor: {metadata.shape}")
            logging.info(f"Demographics tensor: {demographics.shape}")
            logging.info(f"Labels tensor: {labels.shape}")
            
            # Training loop
            n_samples = len(users)
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            for epoch in range(epochs):
                self.model.train()
                total_loss = 0
                
                # Process in batches
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min(start_idx + batch_size, n_samples)
                    
                    batch_users = users[start_idx:end_idx]
                    batch_songs = songs[start_idx:end_idx]
                    batch_metadata = metadata[start_idx:end_idx]
                    batch_demographics = demographics[start_idx:end_idx]
                    batch_labels = labels[start_idx:end_idx]
                    
                    # Log first batch shapes for debugging
                    if epoch == 0 and i == 0:
                        logging.info("First batch tensor shapes:")
                        logging.info(f"Batch users: {batch_users.shape}")
                        logging.info(f"Batch songs: {batch_songs.shape}")
                        logging.info(f"Batch metadata: {batch_metadata.shape}")
                        logging.info(f"Batch demographics: {batch_demographics.shape}")
                        logging.info(f"Batch labels: {batch_labels.shape}")
                    
                    optimizer.zero_grad() # Reset gradient
                    predictions = self.model(
                        batch_users,
                        batch_songs,
                        batch_metadata,
                        batch_demographics
                    )
                    
                    if epoch == 0 and i == 0:
                        logging.info(f"Predictions shape: {predictions.shape}")
                        logging.info(f"Labels shape: {batch_labels.shape}")
                    
                    loss = criterion(predictions, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    if (i + 1) % 10 == 0:
                        logging.info(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{n_batches}, Loss: {loss.item():.4f}")
                
                avg_loss = total_loss / n_batches
                logging.info(f"Epoch {epoch+1}/{epochs} completed - Avg Loss: {avg_loss:.4f}")
                
        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            raise

    async def predict_next_songs(self, user_id: str, demographics: dict, n: int = 5) -> List[Dict]:
        """Predict next n songs for a user using their demographics and song metadata"""
        try:
            # Extract demographic features
            demographic_features = torch.tensor([
                demographics.get('age_group_encoded', 0),
                demographics.get('gender_encoded', 0),
                demographics.get('location_encoded', 0),
                demographics.get('occupation_encoded', 0)
            ], dtype=torch.float32)
            
            # Create a tensor for the user
            user_tensor = torch.tensor([int(self.user_encoder.transform([user_id])[0])], dtype=torch.long)
            
            # Ensure metadata is processed
            if self.metadata_features is None:
                self.process_metadata()
            
            # Generate predictions for all songs
            predictions = []
            batch_size = 32  # Process songs in batches for efficiency
            song_indices = list(range(len(self.song_encoder.classes_)))
            
            for i in range(0, len(song_indices), batch_size):
                batch_indices = song_indices[i:i + batch_size]
                batch_size_actual = len(batch_indices)
                
                # Get metadata for all songs in batch
                metadata_vectors = []
                valid_indices = []
                valid_song_ids = []
                
                for idx, song_idx in enumerate(batch_indices):
                    song_id = int(self.song_encoder.inverse_transform([song_idx])[0])
                    song_metadata = self.get_song_metadata(song_id)
                    
                    if song_metadata is not None:
                        metadata_vectors.append(self._metadata_to_vector(song_metadata))
                        valid_indices.append(idx)
                        valid_song_ids.append(song_id)
                
                if not metadata_vectors:
                    continue
                
                valid_batch_size = len(valid_indices)
                
                # Create tensors for the valid batch
                song_tensors = torch.tensor([batch_indices[i] for i in valid_indices], dtype=torch.long)
                user_tensors = user_tensor.repeat(valid_batch_size)
                demographic_tensors = demographic_features.unsqueeze(0).repeat(valid_batch_size, 1)
                metadata_tensor = torch.tensor(metadata_vectors, dtype=torch.float32)
                
                # Log shapes for debugging
                logging.debug(f"Batch shapes:")
                logging.debug(f"user_tensors: {user_tensors.shape}")
                logging.debug(f"song_tensors: {song_tensors.shape}")
                logging.debug(f"metadata_tensor: {metadata_tensor.shape}")
                logging.debug(f"demographic_tensors: {demographic_tensors.shape}")
                
                # Forward pass
                with torch.no_grad():
                    scores = self.model(
                        user_tensors,
                        song_tensors,
                        metadata_tensor,
                        demographic_tensors
                    )
                
                # Add predictions
                for idx, (song_id, score) in enumerate(zip(valid_song_ids, scores)):
                    song_features = self.get_song_features(song_id)
                    predictions.append({
                        'song_id': str(song_id),
                        'score': float(score.item()),
                        'metadata': {
                            'genres': song_features['genres'],
                            'tags': song_features['tags'],
                            'duration': song_features['duration'],
                            'genre_top': song_features['genre_top']
                        }
                    })
            
            # Sort by score and get top n
            predictions.sort(key=lambda x: x['score'], reverse=True)
            return predictions[:n]
            
        except Exception as e:
            logging.error(f"Error in predict_next_songs: {str(e)}")
            raise

    def process_metadata(self):
        """Process and prepare song metadata features"""
        try:
            # Load tracks data from FMA dataset
            tracks_data = self.fma_processor.load_tracks()
            if tracks_data is None or tracks_data.empty:
                raise ValueError("Failed to load tracks data")
                
            logging.info(f"Loaded {len(tracks_data)} tracks from FMA dataset")
            
            # Process metadata features
            metadata_features = pd.DataFrame(index=tracks_data.index)
            
            # Process genres (convert string representation to list)
            # Process genres (convert string representation to list)
            metadata_features['genres'] = tracks_data['track_genres'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else []
            )
            
            # Process tags (convert string representation to list)  
            metadata_features['tags'] = tracks_data['track_tags'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else []
            )
            
            # Add basic metadata
            metadata_features['duration'] = tracks_data['duration']
            metadata_features['genre_top'] = tracks_data['track_genre_top']
            
            self.metadata_features = metadata_features
            logging.info("Successfully processed metadata features")
            
            return metadata_features
            
        except Exception as e:
            logging.error(f"Error processing metadata: {str(e)}")
            raise
            
    def get_song_metadata(self, song_id):
        """Get processed metadata for a specific song"""
        if self.metadata_features is None:
            self.process_metadata()
            
        if song_id not in self.metadata_features.index:
            logging.warning(f"Song ID {song_id} not found in metadata")
            return None
            
        return self.metadata_features.loc[song_id]
        
    def get_song_features(self, song_id):
        """Get combined song features including metadata"""
        metadata = self.get_song_metadata(song_id)
        if metadata is None:
            return None
            
        features = {
            'genres': metadata['genres'],
            'tags': metadata['tags'],
            'duration': metadata['duration'],
            'genre_top': metadata['genre_top']
        }
        
        return features
