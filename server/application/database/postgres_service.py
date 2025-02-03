import psycopg2
import pandas as pd
import logging
from psycopg2.extras import execute_values
import json
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

# Load environment variables from parent directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), '.env'))

class PostgresService:
    def __init__(self, testing: bool = False):
        try:
            self.conn = psycopg2.connect(
                dbname=os.getenv('POSTGRES_DB', 'music_db'),
                user=os.getenv('POSTGRES_USER', 'music_user'),
                password=os.getenv('POSTGRES_PASSWORD', 'music_password'),
                host=os.getenv('POSTGRES_HOST', 'localhost'),
                port=os.getenv('POSTGRES_PORT', '5433')
            )
            self.cursor = self.conn.cursor()
            logging.info("Successfully connected to PostgreSQL database")
            self._ensure_tables()
        except psycopg2.OperationalError as e:
            logging.error(f"Failed to connect to PostgreSQL: {str(e)}")
            logging.error(f"Using credentials - DB: {os.getenv('POSTGRES_DB')}, User: {os.getenv('POSTGRES_USER')}, Host: {os.getenv('POSTGRES_HOST')}, Port: {os.getenv('POSTGRES_PORT', '5433')}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error connecting to PostgreSQL: {str(e)}")
            raise
        
    def _ensure_tables(self):
        """Ensure all required tables exist"""
        try:
            # Create tracks table if it doesn't exist
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS tracks (
                    id INTEGER PRIMARY KEY,
                    track_title VARCHAR(255) NOT NULL,
                    artist_name VARCHAR(255) NOT NULL,
                    track_genre_top VARCHAR(100),
                    track_genres JSONB,
                    track_tags JSONB,
                    duration FLOAT,
                    album_title VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_tracks_artist ON tracks(artist_name);
                CREATE INDEX IF NOT EXISTS idx_tracks_genre ON tracks(track_genre_top);
                CREATE INDEX IF NOT EXISTS idx_tracks_title ON tracks(track_title);
            """)
            self.conn.commit()
            logging.info("Successfully ensured database tables and indexes")
            
        except Exception as e:
            logging.error(f"Error ensuring tables: {str(e)}")
            self.conn.rollback()
            raise
            
    def store_tracks(self, tracks_df: pd.DataFrame):
        """Store tracks data from DataFrame to PostgreSQL"""
        try:
            # Convert track_genres and track_tags to proper JSON strings
            tracks_df['track_genres'] = tracks_df['track_genres'].apply(
                lambda x: json.dumps(eval(x)) if isinstance(x, str) else json.dumps([])
            )
            tracks_df['track_tags'] = tracks_df['track_tags'].apply(
                lambda x: json.dumps(eval(x)) if isinstance(x, str) else json.dumps([])
            )
            
            # Prepare data for bulk insert
            data = [
                (
                    int(idx),  # id from index
                    row['track_title'],
                    row['artist_name'],
                    row.get('track_genre_top', 'Unknown'),
                    row['track_genres'],
                    row['track_tags'],
                    float(row.get('duration', 0)),
                    row.get('album_title', 'Unknown Album')
                )
                for idx, row in tracks_df.iterrows()
            ]
            
            # Bulk insert using execute_values
            execute_values(
                self.cursor,
                """
                INSERT INTO tracks (
                    id, track_title, artist_name, track_genre_top,
                    track_genres, track_tags, duration, album_title
                )
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    track_title = EXCLUDED.track_title,
                    artist_name = EXCLUDED.artist_name,
                    track_genre_top = EXCLUDED.track_genre_top,
                    track_genres = EXCLUDED.track_genres,
                    track_tags = EXCLUDED.track_tags,
                    duration = EXCLUDED.duration,
                    album_title = EXCLUDED.album_title
                """,
                data
            )
            self.conn.commit()
            
        except Exception as e:
            logging.error(f"Error storing tracks: {str(e)}")
            self.conn.rollback()
            raise
            
    def get_tracks(self, ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Retrieve tracks from PostgreSQL as DataFrame"""
        try:
            query = """
                SELECT 
                    id, track_title, artist_name, track_genre_top,
                    track_genres, track_tags, duration, album_title
                FROM tracks
            """
            
            if ids:
                placeholders = ','.join(['%s'] * len(ids))
                query += f" WHERE id IN ({placeholders})"
                self.cursor.execute(query, ids)
            else:
                self.cursor.execute(query)
                
            columns = ['id', 'track_title', 'artist_name', 'track_genre_top',
                      'track_genres', 'track_tags', 'duration', 'album_title']
            data = self.cursor.fetchall()
            
            df = pd.DataFrame(data, columns=columns)
            if not df.empty:
                df.set_index('id', inplace=True)
                # Convert JSONB strings back to Python objects
                df['track_genres'] = df['track_genres'].apply(json.dumps)
                df['track_tags'] = df['track_tags'].apply(json.dumps)
            return df
            
        except Exception as e:
            logging.error(f"Error retrieving tracks: {str(e)}")
            raise
            
    def search_tracks(self, query: str, limit: int = 5) -> pd.DataFrame:
        """Search tracks by title, artist, or genre"""
        try:
            self.cursor.execute("""
                SELECT 
                    id, track_title, artist_name, track_genre_top,
                    track_genres, track_tags, duration, album_title
                FROM tracks
                WHERE 
                    track_title ILIKE %s OR
                    artist_name ILIKE %s OR
                    track_genre_top ILIKE %s
                LIMIT %s
            """, (f"%{query}%", f"%{query}%", f"%{query}%", limit))
            
            columns = ['id', 'track_title', 'artist_name', 'track_genre_top',
                      'track_genres', 'track_tags', 'duration', 'album_title']
            data = self.cursor.fetchall()
            
            df = pd.DataFrame(data, columns=columns)
            if not df.empty:
                df.set_index('id', inplace=True)
                df['track_genres'] = df['track_genres'].apply(json.dumps)
                df['track_tags'] = df['track_tags'].apply(json.dumps)
            return df
            
        except Exception as e:
            logging.error(f"Error searching tracks: {str(e)}")
            raise
            
    def __del__(self):
        """Clean up database connections"""
        if hasattr(self, 'cursor'):
            self.cursor.close()
        if hasattr(self, 'conn'):
            self.conn.close() 