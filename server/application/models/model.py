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
from collections import defaultdict
from datetime import datetime

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
        
        return min(1.0, max(0.0, confidence)) #Ensure confidence is between 0 and 1
    
    def _calculate_history_factor(self, history_length: int) -> float:
        """
        Calculate confidence factor based on user history length.
        Uses logarithmic scaling to handle varying history lengths.
        """

        """
        log1p(x) = log(1+x), which means that when there are no songs added, log1p(0) = 0, rather than log(0) = -infinity, which'll throw an error.
        divide by log(50) to normalize the history length, because the max history length is 50 songs.
        """

        return min(1.0, np.log1p(history_length) / np.log(50)) # Ensure history factor is between 0 and 1
    
    def _calculate_embedding_factor(
        self,
        predictions: Dict[str, float],
        user_context: Dict[str, Any]
    ) -> float:
        """
        Calculate confidence factor based on embedding similarity.
        Uses normalized cosine similarity between user and song embeddings.
        
        Mathematical formulation:
        1. Normalize embeddings: e_norm = e / ||e|| -> unit vector embeddings
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
            'demographic': None,
            'popularity': None  
        }
        
    def set_model_predictor(self, model_predictor: 'MusicRecommender'):
        """Set the main model predictor."""
        self.prediction_sources['model'] = model_predictor

    def set_demographic_predictor(self, demographic_predictor: 'DemographicPredictor'):
        """Set the demographic predictor."""
        self.prediction_sources['demographic'] = demographic_predictor

    def set_popularity_predictor(self, popularity_predictor: 'PopularityPredictor'):
        """Set the popularity predictor."""
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
            # srouce_names -> Model, Demographic, Popularity
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
            blended_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
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
        
        Mathematical process:
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
    def __init__(self, num_users, num_songs, embedding_dim, demographic_dim):
        super(MusicRecommender, self).__init__()
        
        # Basic embeddings for users and songs
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.song_embedding = nn.Embedding(num_songs, embedding_dim)
        
        # Calculate total input dimension (user embedding + song embedding + demographics)
        total_input_dim = embedding_dim * 2 + demographic_dim
        
        # Simplified network architecture
        self.fc1 = nn.Linear(total_input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)  # Single output for interaction prediction
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, user_input, song_input, demographic_input):
        """Forward pass with simplified inputs"""
        if user_input.dim() == 0:
           user_input = user_input.unsqueeze(0)
        if song_input.dim() == 0:
            song_input = song_input.unsqueeze(0)
        if demographic_input.dim() == 0:
            demographic_input = demographic_input.unsqueeze(0)
        # Get embeddings
        user_embedded = self.user_embedding(user_input)
        song_embedded = self.song_embedding(song_input)
        
        # Handle NaN values in demographics
        demographic_input = torch.nan_to_num(demographic_input, 0.0)
        
        # Concatenate all inputs
        combined = torch.cat((user_embedded, song_embedded, demographic_input), dim=1)
        
        # Forward pass through network
        x = torch.relu(self.fc1(combined))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.output(x))
        
        return x.squeeze()

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
        """Generate training data with only user-song interactions and demographics"""
        try:
            users, songs, demographics, labels = [], [], [], []
            
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
                    users.append(user)
                    songs.append(row['song_encoded'])
                    demographics.append(user_demographics.values)
                    labels.append(1)
                    
                    # Add negative samples (random songs not in user's history)

                    """
                    Negative samples are anti-theses to positive samples: our user(s) never actually listened to these songs, we're adding them to show the Neural Network that, presumably, "this is what the user didn't like."
                    There is one 'negative interaction' for every positive interaction. So, each user has 2x numbers of interactions, if the interactions per user parameter = x. Negative sample songs are taken from the complementary set of songs in listening history
                        set(ns[songs]) = U - {set(listening_history[songs])}    
                            ^negative samples      ^positive samples
                    The labels array is a boolean array (has either 0, 1). 0 -> Negative Sample; 1 -> Positive Sample
                    """
                    user_songs = set(user_history['song_encoded'].values)
                    all_songs = set(range(max(listening_history['song_encoded']) + 1))
                    available_songs = list(all_songs - user_songs)
                    
                    if available_songs:
                        negative_song = int(np.random.choice(available_songs))
                        users.append(user)
                        songs.append(negative_song)
                        demographics.append(user_demographics.values)
                        labels.append(0)
            
            if not users:
                raise ValueError("No training data generated")
                
            logging.info(f"Generated {len(users)} training samples")
            
            return (
                torch.tensor(users),
                torch.tensor(songs),
                torch.tensor(demographics, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.float32)
            )
            
        except Exception as e:
            logging.error(f"Error generating training data: {str(e)}")
            raise

    def fit(self, listening_history, demographics_df, epochs=5, batch_size=32):
        """Train the model with simplified inputs"""
        try:
            # Encode user IDs
            self.user_encoder.fit(listening_history['user_id'].astype(str))
            listening_history['user_encoded'] = self.user_encoder.transform(listening_history['user_id'].astype(str))
            
            # Encode song IDs
            self.song_encoder.fit(listening_history['song_id'].astype(int))
            listening_history['song_encoded'] = self.song_encoder.transform(listening_history['song_id'].astype(int))
            
            # Process demographics
            processed_demographics = self.encode_demographics(demographics_df)
            
            num_users = len(self.user_encoder.classes_)
            num_songs = len(self.song_encoder.classes_)
            
            logging.info(f"Training model with {num_users} users and {num_songs} songs")
            logging.info(f"Using batch size: {batch_size}, epochs: {epochs}")
            
            # Initialize model
            self.model = MusicRecommender(
                num_users,
                num_songs,
                self.embedding_dim,
                self.demographic_dim
            )
            
            optimizer = optim.Adam(self.model.parameters())
            criterion = nn.BCELoss() #Binary Cross Entropy Loss -> [0,1]
            
            # Generate training data
            users, songs, demographics, labels = self.generate_training_data(
                listening_history,
                processed_demographics
            )
            
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
                    batch_demographics = demographics[start_idx:end_idx]
                    batch_labels = labels[start_idx:end_idx]
                    
                    optimizer.zero_grad()
                    predictions = self.model(
                        batch_users,
                        batch_songs,
                        batch_demographics
                    )
                    
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
        """Predict next n songs for a user using their demographics"""
        try:
            # Extract demographic features
            demographic_features = torch.tensor([
                demographics.get('age_group_encoded', 0),
                demographics.get('gender_encoded', 0),
                demographics.get('location_encoded', 0),
                demographics.get('occupation_encoded', 0)
            ], dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            
            # Create a tensor for the user
            user_tensor = torch.tensor([int(self.user_encoder.transform([user_id])[0])], dtype=torch.long)
            
            # Generate predictions for all songs
            predictions = []
            for song_idx in range(len(self.song_encoder.classes_)):
                # Create song tensor
                song_tensor = torch.tensor([song_idx], dtype=torch.long)
                
                # Forward pass
                with torch.no_grad():
                    score = self.model(user_tensor, song_tensor, demographic_features)
                    
                predictions.append({
                    'song_id': str(self.song_encoder.inverse_transform([song_idx])[0]),
                    'score': float(score.item())
                })
            
            # Sort by score and get top n
            predictions.sort(key=lambda x: x['score'], reverse=True)
            return predictions[:n]
            
        except Exception as e:
            logging.error(f"Error in predict_next_songs: {str(e)}")
            raise
