import os
import sys
import logging
from pathlib import Path

# Add the parent directory to the Python path so we can import from application
sys.path.append(str(Path(__file__).parent.parent))

from application.database.postgres_service import PostgresService
from application.models.fma_dataset_processor import FMADatasetProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_database():
    """Initialize PostgreSQL database with FMA dataset"""
    try:
        logger.info("Initializing PostgreSQL database...")
        
        # Initialize services
        pg_service = PostgresService()
        fma_processor = FMADatasetProcessor()
        
        # Process and load the dataset
        logger.info("Processing FMA dataset...")
        tracks_data = fma_processor.process_dataset()
        
        if tracks_data is None or tracks_data.empty:
            raise ValueError("Failed to process FMA dataset")
            
        # Store tracks in PostgreSQL
        logger.info(f"Storing {len(tracks_data)} tracks in PostgreSQL...")
        pg_service.store_tracks(tracks_data)
        
        logger.info("Database initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        init_database()
        logger.info("Database initialization completed successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        sys.exit(1) 