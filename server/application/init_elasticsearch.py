import pandas as pd
from application.services.elasticsearch_service import ElasticsearchService
from application.models.fma_dataset_processor import FMADatasetProcessor

def load_sample_data():
    """Load music data from FMA dataset"""
    # Initialize FMA dataset processor
    fma_processor = FMADatasetProcessor()
    
    # Process and return the dataset
    return fma_processor.process_dataset()

def main():
    # Initialize Elasticsearch service
    es_service = ElasticsearchService()
    
    # Load sample data
    print("Loading sample data...")
    songs_df = load_sample_data()
    
    # Initialize Elasticsearch with the data
    print("Initializing Elasticsearch...")
    success = es_service.initialize_elasticsearch(songs_df)
    
    if success:
        print("Elasticsearch initialization completed successfully!")
    else:
        print("Elasticsearch initialization failed!")

if __name__ == "__main__":
    main() 