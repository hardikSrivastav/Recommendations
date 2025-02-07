import pandas as pd
import os
from pathlib import Path
import logging
from application.database.postgres_service import PostgresService

class FMADatasetProcessor:
    """
    A utility class for processing the Free Music Archive (FMA) dataset metadata.
    """
    
    def __init__(self, base_dir='./fma_metadata'):
        self.base_dir = Path(base_dir)
        self.tracks_path = '/Users/hardiksrivastav/Projects/music/server/fma_metadata/tracks.csv'
        self.genres_path = '/Users/hardiksrivastav/Projects/music/server/fma_metadata/genres.csv'
        self.pg_service = PostgresService()
        
    def load_tracks(self):
        """Loads and processes the tracks metadata."""
        try:
            # First try to get from PostgreSQL
            tracks = self.pg_service.get_tracks()
            if not tracks.empty:
                logging.info("Retrieved tracks from PostgreSQL")
                logging.info(f"Tracks shape: {tracks.shape}")
                logging.info(f"Tracks columns: {tracks.columns.tolist()}")
                logging.info(f"Sample track data:\n{tracks.iloc[0].to_dict()}")
                return tracks
                
            logging.info(f"Loading tracks from {self.tracks_path}")
            
            if not os.path.exists(self.tracks_path):
                raise FileNotFoundError(f"Tracks file not found at {self.tracks_path}")
            
            # Load tracks with multi-level columns
            tracks = pd.read_csv(self.tracks_path, index_col=0, header=[0, 1])
            if tracks.empty:
                raise ValueError("Loaded tracks DataFrame is empty")
                
            logging.info(f"Tracks loaded successfully: {tracks.shape}")
            logging.info(f"Original columns: {tracks.columns.tolist()}")
            
            # Create a new DataFrame with flattened columns
            processed = pd.DataFrame(index=tracks.index)
            
            # Extract relevant columns using multi-level indexing
            try:
                processed['track_title'] = tracks[('track', 'title')]
                processed['artist_name'] = tracks[('artist', 'name')]
                processed['track_genre_top'] = tracks[('track', 'genre_top')]
                processed['track_genres'] = tracks[('track', 'genres')]
                processed['duration'] = tracks[('track', 'duration')]
                processed['track_tags'] = tracks[('track', 'tags')]
                processed['album_title'] = tracks[('album', 'title')]
                
                logging.info("Successfully extracted columns")
                logging.info(f"Processed columns: {processed.columns.tolist()}")
                
            except KeyError as e:
                logging.warning(f"Some columns not found: {e}")
            
            # Fill missing values
            processed = processed.fillna({
                'track_title': 'Unknown Title',
                'artist_name': 'Unknown Artist',
                'track_genre_top': 'Unknown',
                'track_genres': '[]',
                'track_tags': '[]',
                'duration': 0,
                'album_title': 'Unknown Album'
            })
            
            # Convert lists represented as strings to actual lists
            for col in ['track_genres', 'track_tags']:
                processed[col] = processed[col].apply(lambda x: str(x) if isinstance(x, (list, str)) else '[]')
                # Log sample of processed lists
                sample_values = processed[col].head()
                logging.info(f"Sample {col} values:\n{sample_values}")
            
            # Store in PostgreSQL for future use
            self.pg_service.store_tracks(processed)
            logging.info(f"Successfully processed and stored {len(processed)} tracks")
            logging.info(f"Final processed data sample:\n{processed.iloc[0].to_dict()}")
            
            return processed
            
        except Exception as e:
            logging.error(f"Error loading tracks: {str(e)}")
            raise
            
    def process_dataset(self):
        """Process the FMA dataset and return processed tracks data"""
        try:
            # First try to get from PostgreSQL
            tracks = self.pg_service.get_tracks()
            if not tracks.empty:
                logging.info("Retrieved processed dataset from PostgreSQL")
                return tracks
                
            logging.info(f"Loading tracks from {self.tracks_path}")
            
            # Load tracks with specific columns we need
            tracks = pd.read_csv(self.tracks_path, index_col=0, header=[0, 1])
            
            if tracks.empty:
                raise ValueError("No data loaded from tracks.csv")
                
            logging.info(f"Tracks loaded successfully: {tracks.shape}")
            
            # Get the columns we need
            columns_needed = [
                ('track', 'title'),
                ('artist', 'name'),
                ('album', 'title'),
                ('track', 'genre_top'),
                ('track', 'genres'),
                ('track', 'tags'),
                ('track', 'duration')
            ]
            
            # Log available columns for debugging
            logging.info(f"Available columns: {[col for col in columns_needed if col in tracks.columns]}")
            
            # Create a new DataFrame with flattened column names
            processed_tracks = pd.DataFrame()
            processed_tracks.index = tracks.index
            
            # Process each needed column
            if ('track', 'title') in tracks.columns:
                processed_tracks['track_title'] = tracks[('track', 'title')]
            if ('artist', 'name') in tracks.columns:
                processed_tracks['artist_name'] = tracks[('artist', 'name')]
            if ('album', 'title') in tracks.columns:
                processed_tracks['album_title'] = tracks[('album', 'title')]
            if ('track', 'genre_top') in tracks.columns:
                processed_tracks['track_genre_top'] = tracks[('track', 'genre_top')]
            if ('track', 'genres') in tracks.columns:
                processed_tracks['track_genres'] = tracks[('track', 'genres')]
            if ('track', 'tags') in tracks.columns:
                processed_tracks['track_tags'] = tracks[('track', 'tags')]
            if ('track', 'duration') in tracks.columns:
                processed_tracks['duration'] = tracks[('track', 'duration')]
                
            # Fill NaN values
            processed_tracks = processed_tracks.fillna({
                'track_title': 'Unknown Title',
                'artist_name': 'Unknown Artist',
                'track_genre_top': 'Unknown',
                'track_genres': '[]',
                'track_tags': '[]',
                'duration': 0,
                'album_title': 'Unknown Album'
            })
            
            # Store in PostgreSQL for future use
            self.pg_service.store_tracks(processed_tracks)
            logging.info(f"Successfully processed and stored {len(processed_tracks)} tracks")
            
            return processed_tracks
            
        except Exception as e:
            logging.error(f"Error processing FMA dataset: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    None