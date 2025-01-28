import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
        
        # Fully connected (FC) layers for combining embeddings and metadata; each neural network node (neuron) in layer n is connected to every node in layer n-1
        self.fc1 = nn.Linear(total_input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)
        
        # Store dimensions for debugging
        self.embedding_dim = embedding_dim
        self.metadata_dim = metadata_dim
        self.demographic_dim = demographic_dim

    def forward(self, user_input, song_input, metadata_input, demographic_input):
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
        return torch.sigmoid(self.output(x))

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

    def fit(self, listening_history, song_metadata, demographics_df, epochs=10, batch_size=64):
        """Train the model."""
        processed_data, processed_metadata, processed_demographics = self.preprocess_data(
            listening_history, song_metadata, demographics_df
        )
        
        # Important: We need to reindex song_metadata to match the encoded values
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
        criterion = nn.BCELoss() # Gives 0 or 1; loss is low for true positives (TPs) & TNs, high for FPs and FNs 

        # Generate training data 
        users, songs, metadata, demographics, labels = self.generate_training_data(
            processed_data, processed_metadata, processed_demographics
        )

        # Split into train and validation sets
        train_size = int(0.8 * len(users))
        indices = np.arange(len(users))
        np.random.shuffle(indices) # Breaks any patterns in the training data. Avoids the case where all user1 interactions are together, then user2... etc

        train_indices = indices[:train_size] # Training set
        val_indices = indices[train_size:] # Validation set

        users_train, songs_train, metadata_train, demographics_train, labels_train = (
            users[train_indices], songs[train_indices], metadata[train_indices], demographics[train_indices], labels[train_indices]
        )
        users_val, songs_val, metadata_val, demographics_val, labels_val = (
            users[val_indices], songs[val_indices], metadata[val_indices], demographics[val_indices], labels[val_indices]
        )

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad() # Sets the gradient to zero at the beginning of every epoch
            predictions = self.model(users_train, songs_train, metadata_train, demographics_train).squeeze() # Squeeze because BCE predictions are in [batch_size, 1] shape, i.e. each value in the present unsqueezed array is a vector in itself.
            loss = criterion(predictions, labels_train)
            loss.backward()
            optimizer.step()

            self.model.eval() # Evaluates the performance of the model
            with torch.no_grad(): # Don't need to evaluate gradients during validation
                val_predictions = self.model(users_val, songs_val, metadata_val, demographics_val).squeeze()
                val_loss = criterion(val_predictions, labels_val)

            logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    def predict_next_songs(self, user_id,  track_data, demographics_df, n=5):
        """Predict next songs for a user."""
        encoded_demographics = self.encode_demographics(demographics_df)
        user_encoded = self.user_encoder.transform([user_id])[0] # user_0 -> proxy for user_0 (refer line 124)
        all_songs = torch.arange(len(self.song_encoder.classes_)) # basically preparing input data. Generating an array of user proxies of length = all songs in the dataset so that we can stack-rank all songs
        
        # Generate metadata tensor for all songs
        metadata = torch.zeros((len(all_songs), self.metadata_dim)) # bunch of zeros matching the array size of all songs
        
        user_demographics = encoded_demographics[encoded_demographics['user_id'] == user_id].iloc[0]
        demographic_tensor = torch.tensor([
            user_demographics['age_group_encoded'],
            user_demographics['gender_encoded'],
            user_demographics['location_encoded'],
            user_demographics['occupation_encoded']
        ], dtype=torch.float32).repeat(len(all_songs), 1)

        user_input = torch.full((len(all_songs),), user_encoded)
        logging.debug(f"Generating predictions for user {user_id}")
        
        with torch.no_grad():
            predictions = self.model(
                user_input, 
                all_songs, 
                metadata,
                demographic_tensor
            ).squeeze() # predicting the likelihood that the user will like all songs (0 < likelihood < 1, due to BCE)
        
        # Adjust n to be no larger than the number of available songs
        n = min(n, len(predictions))
        logging.info(f"Requesting {n} recommendations from {len(predictions)} available songs")
        
        top_n_indices = torch.topk(predictions, n).indices # Selecting the song proxies for the top n songs from our list of all songs
        recommended_ids = list(self.song_encoder.inverse_transform(top_n_indices.numpy()))
        song_details = []
        for track_id in recommended_ids:
            track = track_data.loc[track_id]
            song_details.append(f"{track['track_title']} - {track['artist_name']}") # Song Title - Artist Name format
        return song_details

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
