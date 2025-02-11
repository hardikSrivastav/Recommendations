#!/usr/bin/env python3
import os
import sys
import click
from pymongo import MongoClient
import redis
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Add the server directory to Python path
SERVER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

try:
    from application.database.database_service import Base
except ImportError:
    print("Error: Cannot import application modules.")
    print("Current Python path:", sys.path)
    print("Try running this script from the server directory using:")
    print("  PYTHONPATH=$PYTHONPATH:/path/to/server python -m application.database.manage_db init")
    sys.exit(1)

# Database URLs
POSTGRES_URL = "postgresql://music_user:music_password@localhost:5433/music_db"
MONGO_URL = "mongodb://localhost:27017/music_db"
REDIS_URL = "redis://:music_password@localhost:6380/0"

@click.group()
def cli():
    """Database management commands."""
    pass

@cli.command()
def view_demographics():
    """View all user demographics from PostgreSQL."""
    click.echo("Fetching user demographics from PostgreSQL...")
    engine = create_engine(POSTGRES_URL)
    
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT 
                    user_id,
                    age,
                    gender,
                    location,
                    occupation,
                    created_at,
                    updated_at
                FROM user_demographics
                ORDER BY created_at DESC
            """)
            
            result = conn.execute(query)
            
            # Convert to list to get row count
            rows = list(result)
            
            if not rows:
                click.echo("No user demographics found.")
                return
                
            click.echo(f"\nFound {len(rows)} user(s):\n")
            
            # Print each row in a formatted way
            for row in rows:
                click.echo("-" * 50)
                click.echo(f"User ID: {row.user_id}")
                click.echo(f"Age: {row.age}")
                click.echo(f"Gender: {row.gender}")
                click.echo(f"Location: {row.location}")
                click.echo(f"Occupation: {row.occupation}")
                click.echo(f"Created: {row.created_at}")
                click.echo(f"Updated: {row.updated_at}")
            
            click.echo("-" * 50)
            
    except SQLAlchemyError as e:
        click.echo(f"Database error: {str(e)}")
        raise
    except Exception as e:
        click.echo(f"Error fetching demographics: {str(e)}")
        raise

@cli.command()
def migrate():
    """Migrate database schema to latest version."""
    click.echo("Migrating PostgreSQL schema...")
    engine = create_engine(POSTGRES_URL)
    
    # Drop and recreate the user_demographics table
    with engine.connect() as conn:
        conn.execute("DROP TABLE IF EXISTS user_demographics CASCADE")
        conn.execute("""
            CREATE TABLE user_demographics (
                user_id VARCHAR(50) PRIMARY KEY,
                age INTEGER NOT NULL,
                gender VARCHAR(10) NOT NULL,
                location VARCHAR(2) NOT NULL,
                occupation VARCHAR(50) NOT NULL,
                created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)
    click.echo("Migration completed successfully.")

@cli.command()
def init():
    """Initialize all databases with required schemas and indexes."""
    # PostgreSQL initialization
    click.echo("Initializing PostgreSQL...")
    engine = create_engine(POSTGRES_URL)
    Base.metadata.create_all(engine)
    click.echo("PostgreSQL initialized successfully.")

    # MongoDB initialization
    click.echo("Initializing MongoDB...")
    client = MongoClient(MONGO_URL)
    db = client.music_db

    # Create indexes
    db.listening_history.create_index([("user_id", 1)])
    db.listening_history.create_index([("timestamp", 1)])
    db.feedback.create_index([("user_id", 1)])
    db.feedback.create_index([("timestamp", 1)])
    db.predictions.create_index([("user_id", 1), ("timestamp", 1)])
    click.echo("MongoDB initialized successfully.")

    # Redis initialization
    click.echo("Initializing Redis...")
    redis_client = redis.from_url(REDIS_URL)
    redis_client.ping()  # Test connection
    click.echo("Redis initialized successfully.")

@cli.command()
def reset():
    """Reset all databases (WARNING: This will delete all data)."""
    if not click.confirm("This will delete all data. Are you sure?"):
        return

    # Reset PostgreSQL
    click.echo("Resetting PostgreSQL...")
    engine = create_engine(POSTGRES_URL)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    click.echo("PostgreSQL reset successfully.")

    # Reset MongoDB
    click.echo("Resetting MongoDB...")
    client = MongoClient(MONGO_URL)
    client.drop_database('music_db')
    click.echo("MongoDB reset successfully.")

    # Reset Redis
    click.echo("Resetting Redis...")
    redis_client = redis.from_url(REDIS_URL)
    redis_client.flushdb()
    click.echo("Redis reset successfully.")

@cli.command()
def check():
    """Check database connections and schemas."""
    # Check PostgreSQL
    click.echo("Checking PostgreSQL...")
    try:
        engine = create_engine(POSTGRES_URL)
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        click.echo("✅ PostgreSQL connection successful")
    except Exception as e:
        click.echo(f"❌ PostgreSQL error: {str(e)}")

    # Check MongoDB
    click.echo("Checking MongoDB...")
    try:
        client = MongoClient(MONGO_URL)
        client.admin.command('ping')
        click.echo("✅ MongoDB connection successful")
    except Exception as e:
        click.echo(f"❌ MongoDB error: {str(e)}")

    # Check Redis
    click.echo("Checking Redis...")
    try:
        redis_client = redis.from_url(REDIS_URL)
        redis_client.ping()
        click.echo("✅ Redis connection successful")
    except Exception as e:
        click.echo(f"❌ Redis error: {str(e)}")

if __name__ == '__main__':
    cli() 