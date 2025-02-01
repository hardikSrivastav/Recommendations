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
        - f_h = min(1, log(1 + history_length) / log(50))  # Normalized history factor
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
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_history_factor(self, history_length: int) -> float:
        """
        Calculate confidence factor based on user history length.
        Uses logarithmic scaling to handle varying history lengths.
        """
        return min(1.0, np.log1p(history_length) / np.log(50))
    
    def _calculate_embedding_factor(
        self,
        predictions: Dict[str, float],
        user_context: Dict[str, Any]
    ) -> float:
        """
        Calculate confidence factor based on embedding similarity.
        Uses normalized cosine similarity between user and song embeddings.
        
        Mathematical formulation:
        1. Normalize embeddings: e_norm = e / ||e||
        2. Calculate cosine similarity: cos_sim = (u_norm · s_norm)
        3. Scale to [0,1]: similarity = (cos_sim + 1) / 2
        where:
        - e is the embedding vector
        - ||e|| is the L2 norm of the vector
        - u_norm is the normalized user embedding
        - s_norm is the normalized song embedding
        - · represents dot product
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
            'demographic': None,  # Will be implemented
            'popularity': None  # Will be implemented
        }
        
    def set_model_predictor(self, model_predictor: 'MusicRecommender'):
        """Set the main model predictor."""
        self.prediction_sources['model'] = model_predictor
        
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
        if hasattr(predictor, 'predict_async'):
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
    def __init__(self, num_users, num_songs, embedding_dim, metadata_dim, demographic_dim):
        super(MusicRecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.song_embedding = nn.Embedding(num_songs, embedding_dim)
        
        # Calculate total input dimension
        '''
        total input dim = 
            user data embedding layers (user interaction with music) + 
            song data embedding layers (song features and listening patterns) + 
            metadata encoding layers (genres, tags, etc)
            demographic encoding layers (age, location, occupation, gender)
        user_data dimensions = song_data dimensions = embedding dimensions (embedding_dim)    
        '''
        total_input_dim = embedding_dim * 2 + metadata_dim + demographic_dim
        
        # Fully connected (FC) layers for combining embeddings and metadata
        self.fc1 = nn.Linear(total_input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 2)  # Changed to output [score, confidence]
        
        # Store dimensions for debugging
        self.embedding_dim = embedding_dim
        self.metadata_dim = metadata_dim
        self.demographic_dim = demographic_dim

    def forward(self, user_input, song_input, metadata_input, demographic_input):
        """
        Forward pass with confidence score calculation.
        Returns both prediction score and model's confidence.
        
        Mathematical formulation:
        1. Embedding concatenation: E = [E_user; E_song; M; D]
        2. Forward propagation: h_i = ReLU(W_i * h_{i-1} + b_i)
        3. Output: [score, confidence] = sigmoid(W_out * h_3 + b_out)
        where:
        - E_user, E_song are embedding vectors
        - M is metadata vector
        - D is demographic vector
        - h_i are hidden layer outputs
        """
        metadata_input = torch.nan_to_num(metadata_input, 0.0)
        demographic_input = torch.nan_to_num(demographic_input, 0.0) # If values are NaN, replace with 0
        user_embedded = self.user_embedding(user_input)
        song_embedded = self.song_embedding(song_input)

        # Get batch size and ensure all tensors have correct batch dimension
        batch_size = user_embedded.size(0)
        
        # Ensure metadata has correct shape (batch_size x metadata_dim)
        """
        If metadata arrives in any other shape than batch_size (batch_size = i, if user_embeddings is a matrix of i * j) * metadata_dimension (pre-determined), 
        then we'll have to collapse the array in size batch_size * metadata_dimension to ensure proper matrix multiplication. Same way for demographic details.
        """

        if metadata_input.dim() == 3:
            metadata_input = metadata_input.view(batch_size, -1)
        elif metadata_input.dim() == 1:
            metadata_input = metadata_input.view(batch_size, -1)

        if demographic_input.dim() == 3:
            demographic_input = demographic_input.view(batch_size, -1)
        elif demographic_input.dim() == 1:
            demographic_input = demographic_input.view(batch_size, -1)
            
        # Log shapes for debugging
        logging.debug(f"User embedded shape: {user_embedded.shape}")
        logging.debug(f"Song embedded shape: {song_embedded.shape}")
        logging.debug(f"Metadata shape: {metadata_input.shape}")
        logging.debug(f"Demographic shape: {demographic_input.shape}")
        
        # Concatenate along the feature dimension

        """
        Capturing everything about the data in one tensor to put in the Neural Network. 
            Song Embedding data is in Tensor 1 (T1) containing the data of n song-user interactions, for k no. of embedding dimensions (EDs) for each song. dim(T1) = n*k.
            User Embedding data is in T2 containing the data of n user-song interactions, for k no. of EDs for each song. dim(T2) = n*k
            Metadata for each of the n songs and users (not song-user or user-song interaction) is in T3, with m metadata dimensions (tracks, genres,... m). dim(T3) = n*m
            Demographic data for each of the n users is in T4, with l demographic dimensions (age, location, gender, occupation,... l). dim(T4) = n*l
        Combined tensor (T5): Concatenation across dim 1 (as opposed to dim 0) means that we're horizontally combining our tensors (0 = rows; 1 = columns). So, dim(T4) = n*(k+k+m+l) = n*(2k+m+l)
        Here, n = batch_size, 2k+m = total_embedding_dimensions, k = song_embedding_dimensions = user_embedding_dimensions, m = metadata dimensions
        """
        combined = torch.cat((user_embedded, song_embedded, metadata_input, demographic_input), dim=1)
        logging.debug(f"Combined shape: {combined.shape}")
        
        x = torch.relu(self.fc1(combined))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        output = torch.sigmoid(self.output(x))
        
        # Split output into score and confidence
        score, confidence = output.split(1, dim=1)
        return score.squeeze(), confidence.squeeze()

    def get_user_embedding(self, user_input):
        """Get user embedding for a given user input."""
        return self.user_embedding(user_input)

    def get_song_embedding(self, song_input):
        """Get song embedding for a given song input."""
        return self.song_embedding(song_input)

class RecommenderSystem:
    def __init__(self, embedding_dim=50, metadata_dim=10, demographic_dim=4):
        self.embedding_dim = embedding_dim
        self.metadata_dim = metadata_dim
        self.demographic_dim = demographic_dim
        self.user_encoder = LabelEncoder()
        self.song_encoder = LabelEncoder()
        self.genre_encoder = LabelEncoder()
        self.encoders = {
            'age_group': LabelEncoder(),
            'gender': LabelEncoder(),
            'location': LabelEncoder(),
            'occupation': LabelEncoder()
        }
        self.model = None
        self.ensemble = WeightedEnsembleRecommender()
        self.confidence_weight = 0.3  # Weight for confidence loss

    def get_user_context(self, user_id: str, demographics_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get user context including history length, embeddings, and demographics.
        This context is used for confidence calculation and prediction blending.
        """
        try:
            context = {
                'user_id': user_id,
                'history_length': 0,  # Will be updated when history is available
                'demographics': {},
                'song_embeddings': {}  # Will store song embeddings for similarity calculation
            }
            
            # Add demographic information
            if user_id in demographics_df['user_id'].values:
                user_demographics = demographics_df[demographics_df['user_id'] == user_id].iloc[0]
                context['demographics'] = {
                    'age_group': user_demographics['age_group_encoded'],
                    'gender': user_demographics['gender_encoded'],
                    'location': user_demographics['location_encoded'],
                    'occupation': user_demographics['occupation_encoded']
                }
                
            # Add embeddings if model is trained
            if self.model is not None:
                user_encoded = self.user_encoder.transform([user_id])[0]
                user_input = torch.tensor([user_encoded])
                with torch.no_grad():
                    context['user_embedding'] = self.model.get_user_embedding(user_input).squeeze()
                    
                    # Pre-compute song embeddings for all songs
                    all_songs = torch.arange(len(self.song_encoder.classes_))
                    song_embeddings = self.model.get_song_embedding(all_songs)
                    context['song_embeddings'] = {
                        str(i): emb for i, emb in enumerate(song_embeddings)
                    }
                    
            return context
        except Exception as e:
            logging.error(f"Error getting user context for {user_id}: {str(e)}")
            raise

    async def predict_next_songs(self, user_id: str, track_data: pd.DataFrame, demographics_df: pd.DataFrame, n: int = 5) -> List[Dict[str, Any]]:
        """
        Predict next songs using the ensemble system.
        Returns recommendations with confidence scores and source weights.
        
        Args:
            user_id: The ID of the user to generate recommendations for
            track_data: DataFrame containing track metadata
            demographics_df: DataFrame containing user demographics
            n: Number of recommendations to return

        Returns:
            List of dictionaries containing song details, scores, and confidence values
        
        Raises:
            ValueError: If model is not trained
            KeyError: If user_id not found in demographics
            Exception: For other unexpected errors
        """
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model needs to be trained first")
            
        try:
            # Get user context
            user_context = self.get_user_context(user_id, demographics_df)
            
            # Set up ensemble
            self.ensemble.set_model_predictor(self.model)
            
            # Get recommendations from ensemble
            recommendations = await self.ensemble.get_recommendations(
                user_id,
                user_context,
                n=n
            )
            
            # Format recommendations with track details
            detailed_recommendations = []
            for rec in recommendations:
                track_id = self.song_encoder.inverse_transform([int(rec['song_id'])])[0]
                track = track_data.loc[track_id]
                detailed_recommendations.append({
                    'song': f"{track['track_title']} - {track['artist_name']}",
                    'score': float(rec['score']),
                    'confidence': float(rec['confidence']),
                    'source_weights': rec['source_weights']
                })
                
            return detailed_recommendations
            
        except KeyError as e:
            logging.error(f"User {user_id} not found in demographics data: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error generating predictions for user {user_id}: {str(e)}")
            raise

    def encode_demographics(self, demographics_df):
        """Encode demographic features"""
        encoded_df = demographics_df.copy()
        
        # Create age groups
        encoded_df['age_group'] = pd.cut(encoded_df['age'], 
                                       bins=[0, 18, 25, 35, 50, 100],
                                       labels=['<18', '18-25', '26-35', '36-50', '50+'])
        
        """
        The LabelEncoder() class is used to encode the categorical data into model-understandable numerical data.
        The pd.cut function would basically look a continuous array of numbers and look at the the categories (bins) that they fall into. 
        This basically turns a continuous array into a categorical array (can visualise this using a histogram).            
        """
        
        # Encode each demographic feature
        for col in self.encoders.keys():
            if col in encoded_df.columns:
                encoded_df[f'{col}_encoded'] = self.encoders[col].fit_transform(encoded_df[col])
        
        """
        The fit_transform:
        1. looks at the initial raw data and 'fits' it into the Label Encoder (line 100). This encoder (and sets of encoders) create a hashmap (map) of present raw initial data and their intended 'encoded' numerical values.
        2. This mapping itself is stored inside the function as a class; no edits have been made to the dataset yet. It has only been fit into the encoder.
        3. The 'transform' function now picks up this mapping from within the code and essentially transforms the raw data into the numerical data.
        4. Returns the transformed (numerical) data as a tensor.
        """

        return encoded_df

    def preprocess_data(self, listening_history, song_metadata, demographics_df):
        """
        Preprocess listening history and metadata.
        listening_history: DataFrame with columns [user_id, song_id, timestamp].
        song_metadata: DataFrame with columns [song_id, metadata fields...].
        """
        # First, filter metadata to only include songs that exist in listening_history -> remove songs present in our dataset but songs that none of our users has listened to.
        valid_song_ids = list(listening_history['song_id'].unique())
        song_metadata = song_metadata[song_metadata['song_id'].isin(valid_song_ids)].copy()
        
        # Encode user IDs
        self.user_encoder.fit(listening_history['user_id'].unique())
        listening_history['user_encoded'] = self.user_encoder.transform(listening_history['user_id'])
        
        # Encode song IDs
        self.song_encoder.fit(valid_song_ids)
        listening_history['song_encoded'] = self.song_encoder.transform(listening_history['song_id'])
        song_metadata['song_encoded'] = self.song_encoder.transform(song_metadata['song_id'])
        
        # Process metadata (e.g., genres, tags)
        for col in ['tags', 'genres']:
            if col in song_metadata.columns:
                song_metadata[col] = song_metadata[col].fillna('').apply(
                    lambda x: ','.join(sorted(set(str(x).split(','))))
                )
                self.genre_encoder.fit(song_metadata[col].unique())
                song_metadata[col] = self.genre_encoder.transform(song_metadata[col])

        """
        Let's say x = "rock,pop,rock,indie,pop"
        
        code: str(x).split(',')  
        output: ['rock', 'pop', 'rock', 'indie', 'pop']

        code: set(str(x).split(','))  
        output: {'rock', 'pop', 'indie'} -> Removes duplicates

        sorted(set(str(x).split(',')))  
        ['indie', 'pop', 'rock']  -> Alphabetically sorted

        ','.join(['indie', 'pop', 'rock'])  
        "indie,pop,rock" -> Final result

        The lambda function applies this throughout the metadata dataset
        """

        """
        The data that we've selected constitutes 100% of our 'positive examples', i.e. examples of actual interactions between humans and songs -> our users have listened to these songs.
        We've encoded the data for each of these interactions, i.e. relationship between user and songs, into the user's and the song's song_id, which acts as a proxy for the name of the user and the title of the song. Similarly for metadata
        """
        encoded_demographics = self.encode_demographics(demographics_df)

        return listening_history, song_metadata, encoded_demographics

    def generate_training_data(self, listening_history, song_metadata, demographics_df):
        """Generate training data with metadata for input."""
        users, songs, metadata, demographics, labels = [], [], [], [], []
        
        # Merge metadata and the listening history (user-song interactions) at the start
        listening_history = listening_history.merge(song_metadata, on='song_encoded', how='left')
        listening_history = listening_history.merge(
            demographics_df, 
            left_on='user_id', 
            right_on='user_id', 
            how='left'
        )
        
        # Get valid song indices
        valid_song_indices = song_metadata['song_encoded'].values

        for user in listening_history['user_encoded'].unique(): # Go user-by-user
            # Segregate the chosen user's listening history from the cumulative listening history + metadata matrix/tensor            
            user_history = listening_history[listening_history['user_encoded'] == user]
            # Segregate the chosen user's demographic details from the cumulative demographic details matrix/tensor
            user_demographics = user_history[['age_group_encoded', 'gender_encoded', 'location_encoded', 'occupation_encoded']].iloc[0]
            
            # Positive samples
            for _, row in user_history.iterrows():
                users.append(user)
                songs.append(row['song_encoded'])
                # Ensure metadata is 2D
                meta_values = row[['tags', 'genres']].fillna(0).values.astype(float).flatten()
                metadata.append(meta_values)
                demographics.append(user_demographics.values)
                labels.append(1)
                
                # Negative sample with actual metadata
                """
                Negative samples are anti-theses to positive samples: our user(s) never actually listened to these songs, we're adding them to show the Neural Network that, presumably, "this is what the user didn't like."
                There is one 'negative interaction' for every positive interaction. So, each user has 2x numbers of interactions, if the interactions per user parameter = x. Negative sample songs are taken from the complementary set of songs in listening history
                    set(ns[songs]) = U - {set(listening_history[songs])}    
                          ^negative samples      ^positive samples
                The labels array is a boolean array (has either 0, 1). 0 -> Negative Sample; 1 -> Positive Sample
                """
                negative_song_idx = np.random.choice(len(valid_song_indices))
                while valid_song_indices[negative_song_idx] in user_history['song_encoded'].values:
                    negative_song_idx = np.random.choice(len(valid_song_indices))
                    
                negative_song = valid_song_indices[negative_song_idx]
                users.append(user)
                songs.append(negative_song)
                # Ensure metadata is 2D for negative samples too
                meta_values = song_metadata.iloc[negative_song_idx][['tags', 'genres']].fillna(0).values.astype(float).flatten()
                metadata.append(meta_values)
                demographics.append(user_demographics.values)
                labels.append(0)

        return (torch.tensor(users), torch.tensor(songs), 
                torch.tensor(metadata, dtype=torch.float32),
                torch.tensor(demographics, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.float32))

    def _calculate_combined_loss(self, predictions: torch.Tensor, confidence: torch.Tensor, 
                               labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss incorporating both prediction accuracy and confidence.
        
        Mathematical formulation:
        L = (1 - w) * BCE(pred, label) + w * L_conf
        where:
        - w is the confidence weight
        - BCE is binary cross entropy
        - L_conf = |conf - acc| where acc is 1 if pred matches label, 0 otherwise
        """
        # Calculate prediction loss (BCE)
        criterion = nn.BCELoss()
        pred_loss = criterion(predictions, labels)
        
        # Calculate accuracy for each prediction
        with torch.no_grad():
            accuracy = (predictions.round() == labels).float()
        
        # Calculate confidence loss (how well confidence predicts accuracy)
        conf_loss = torch.abs(confidence - accuracy).mean()
        
        # Combine losses
        total_loss = (1 - self.confidence_weight) * pred_loss + self.confidence_weight * conf_loss
        
        return total_loss

    def fit(self, listening_history, song_metadata, demographics_df, epochs=10, batch_size=64):
        """
        Train the model with enhanced monitoring and confidence calculation.
        
        The training process now optimizes for both prediction accuracy and 
        confidence estimation using a combined loss function.
        """
        processed_data, processed_metadata, processed_demographics = self.preprocess_data(
            listening_history, song_metadata, demographics_df
        )
        
        processed_metadata.index = range(len(processed_metadata))
        
        num_users = len(self.user_encoder.classes_)
        num_songs = max(processed_data['song_encoded']) + 1 # +1 since indexing starts at 0; the maximum index label will always give (len-1)

        logging.info(f"Initializing model with {num_users} users and {num_songs} songs")

        # Initialise model
        self.model = MusicRecommender(
            num_users, num_songs, 
            self.embedding_dim, 
            self.metadata_dim,
            self.demographic_dim
        )
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        users, songs, metadata, demographics, labels = self.generate_training_data(
            processed_data, processed_metadata, processed_demographics
        )

        # Split data into training and validation sets
        train_size = int(0.8 * len(users))
        indices = np.arange(len(users))
        np.random.shuffle(indices) # Breaks any patterns in the training data. Avoids the case where all user1 interactions are together, then user2... etc

        train_indices = indices[:train_size] # Training set
        val_indices = indices[train_size:] # Validation set

        users_train, songs_train, metadata_train, demographics_train, labels_train = (
            users[train_indices], songs[train_indices], metadata[train_indices],
            demographics[train_indices], labels[train_indices]
        )
        users_val, songs_val, metadata_val, demographics_val, labels_val = (
            users[val_indices], songs[val_indices], metadata[val_indices],
            demographics[val_indices], labels[val_indices]
        )

        best_val_loss = float('inf')
        patience = 5  # Number of epochs to wait for improvement
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            # Get predictions and confidence scores
            predictions, confidence = self.model(
                users_train, songs_train,
                metadata_train, demographics_train
            )
            
            # Calculate combined loss
            loss = self._calculate_combined_loss(predictions, confidence, labels_train)
            loss.backward()
            optimizer.step()

            # Validation step
            self.model.eval()
            with torch.no_grad():
                val_predictions, val_confidence = self.model(
                    users_val, songs_val,
                    metadata_val, demographics_val
                )
                val_loss = self._calculate_combined_loss(val_predictions, val_confidence, labels_val)

                # Calculate metrics for logging
                train_accuracy = (predictions.round() == labels_train).float().mean()
                val_accuracy = (val_predictions.round() == labels_val).float().mean()
                
                # Calculate confidence calibration (how well confidence predicts accuracy)
                train_conf_error = torch.abs(confidence.mean() - train_accuracy)
                val_conf_error = torch.abs(val_confidence.mean() - val_accuracy)

            # Log detailed metrics
            logging.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {loss.item():.4f}, "
                f"Val Loss: {val_loss.item():.4f}, "
                f"Train Acc: {train_accuracy.item():.4f}, "
                f"Val Acc: {val_accuracy.item():.4f}, "
                f"Train Conf Error: {train_conf_error.item():.4f}, "
                f"Val Conf Error: {val_conf_error.item():.4f}"
            )

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

    def save_model(self, filepath):
        """Save the model and encoders."""
        torch.save(self.model.state_dict(), f"{filepath}_model.pt")
        with open(f"{filepath}_encoders.pkl", 'wb') as f:
            pickle.dump({
                'user_encoder': self.user_encoder,
                'song_encoder': self.song_encoder,
                'genre_encoder': self.genre_encoder,
                'demographic_encoders': self.encoders,
            }, f)

    def load_model(self, filepath, num_users, num_songs):
        """Load the model and encoders."""
        self.model = MusicRecommender(num_users, num_songs, self.embedding_dim, self.metadata_dim, self.demographic_dim)
        self.model.load_state_dict(torch.load(f"{filepath}_model.pt"))

        with open(f"{filepath}_encoders.pkl", 'rb') as f:
            encoders = pickle.load(f)
            self.user_encoder = encoders['user_encoder']
            self.song_encoder = encoders['song_encoder']
            self.genre_encoder = encoders['genre_encoder']
            self.encoders = encoders['demographic_encoders']
