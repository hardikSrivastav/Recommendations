import pandas as pd
import os
from pathlib import Path
import logging

class FMADatasetProcessor:
    """
    A utility class for processing the Free Music Archive (FMA) dataset metadata.
    """
    
    def __init__(self, base_dir='./fma_metadata'):
        self.base_dir = Path(base_dir)
        self.tracks_path = '/Users/hardiksrivastav/Projects/music/server/fma_metadata/tracks.csv'
        self.genres_path = '/Users/hardiksrivastav/Projects/music/server/fma_metadata/genres.csv'
        
    def load_tracks(self):
        """Loads and processes the tracks metadata."""
        try:
            # Load tracks with multi-level columns
            tracks = pd.read_csv(self.tracks_path, index_col=0, header=[0, 1])
            print(f"Tracks loaded successfully: {tracks.shape}")
            
            # Select subset of columns
            keep_cols = [
                ('track', 'title'),
                ('artist', 'name'),
                ('album', 'title')
            ]
            
            # Rename columns to match expected format
            tracks = tracks[keep_cols]
            tracks.columns = ['track_title', 'artist_name', 'album_title']
            
            return tracks
            
        except Exception as e:
            print(f"Error loading tracks: {e}")
            return None
    
    def process_dataset(self):
        """
        Process the complete dataset and return a clean DataFrame
        with essential music metadata.
        """
        try:
            tracks = self.load_tracks()
            if tracks is None:
                print("Failed to load tracks data")
                return None
                
            # Clean up the data
            tracks = tracks.fillna({
                'track_title': 'Unknown Title',
                'artist_name': 'Unknown Artist',
                'album_title': 'Unknown Album'
            })
            
            return tracks
            
        except Exception as e:
            print(f"Error processing dataset: {e}")
            return None

# Example usage
if __name__ == "__main__":
    None