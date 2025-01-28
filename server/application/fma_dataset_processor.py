import pandas as pd
import os
import requests
from pathlib import Path

class FMADatasetProcessor:
    """
    A utility class for downloading and processing the Free Music Archive (FMA) dataset metadata.
    """
    
    def __init__(self, base_dir='./fma_metadata'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.metadata_url = 'https://os.unil.cloud.switch.ch/fma/fma_metadata.zip'

        """

    def download_metadata(self):
        #Downloads the metadata if it doesn't exist locally
        zip_path = self.base_dir / 'fma_metadata.zip'
        
        if not zip_path.exists():
            print("Downloading metadata...")
            response = requests.get(self.metadata_url, stream=True)
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Unzip the file
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.base_dir)

        """
                
    def load_tracks(self):
        """Loads and processes the tracks metadata."""
        tracks_path = '/Users/hardiksrivastav/Projects/music/server/fma_metadata/tracks.csv'
            
        # Load tracks with multi-level columns
        try:
            tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
            #print(f"Tracks loaded successfully: {tracks.shape}")
        except Exception as e:
            print(f"Error loading tracks: {e}")
            return None

        
        # Select subset of columns
        keep_cols = [
            ('track', 'tags'), ('track', 'genres'),
            ('album', 'title'), ('album', 'date_created'),
            ('artist', 'name'), ('artist', 'location'),
            ('track', 'date_created'), ('track', 'title')
        ]
        
        return tracks[keep_cols]
    
    def get_genre_mapping(self):
        """Loads the genre mapping."""
        genres_path = '/Users/hardiksrivastav/Projects/music/server/fma_metadata/genres.csv'
        
        try:    
            genres = pd.read_csv(genres_path, index_col=0)
            #print(f"Genres loaded successfully: {genres.shape}")
            return genres
        except Exception as e:
            print(f"Error loading genres: {e}")
            return None
    
    def process_dataset(self):
        """
        Process the complete dataset and return a clean DataFrame
        with essential music metadata.
        """
        tracks = self.load_tracks()
        genres = self.get_genre_mapping()
        
        # Flatten multi-level columns
        tracks.columns = [f"{x}_{y}" for x, y in tracks.columns]
        
        # Clean up the data
        cleaned_tracks = tracks.copy()
        cleaned_tracks['album_date_created'] = pd.to_datetime(
            cleaned_tracks['album_date_created']
        )
        cleaned_tracks['track_date_created'] = pd.to_datetime(
            cleaned_tracks['track_date_created']
        )
        
        return cleaned_tracks
    
    def get_sample_dataset(self, n=1000):
        """Returns a sample of the processed dataset."""
        full_dataset = self.process_dataset()
        return full_dataset.sample(n=min(n, len(full_dataset)))

# Example usage
if __name__ == "__main__":
    None