from application.database.database_service import DatabaseService
from datetime import datetime
from pymongo import MongoClient

def migrate_feedback_to_history():
    """
    Migrate incorrectly stored history entries from feedback to listening_history collection
    """
    # Initialize database service
    db_service = DatabaseService()
    
    try:
        # Get all entries from feedback collection
        feedback_entries = list(db_service.mongodb.feedback.find({}))
        print(f"Found {len(feedback_entries)} entries in feedback collection")
        
        # Move each entry to listening_history with proper format
        for entry in feedback_entries:
            history_data = {
                'user_id': str(entry['user_id']),
                'song_id': int(entry['song_id']),
                'timestamp': entry['timestamp'],
                'source': 'manual_add_migrated',  # Mark these as migrated entries
                'duration_seconds': None,
                'completed': True
            }
            
            # Insert into listening_history
            db_service.mongodb.listening_history.insert_one(history_data)
            
            # Remove from feedback collection
            db_service.mongodb.feedback.delete_one({'_id': entry['_id']})
            
            print(f"Migrated entry for user {entry['user_id']}, song {entry['song_id']}")
            
        print("\nMigration completed successfully!")
        print(f"Moved {len(feedback_entries)} entries from feedback to listening_history")
        
    except Exception as e:
        print(f"Error during migration: {str(e)}")
        raise e

if __name__ == "__main__":
    migrate_feedback_to_history() 