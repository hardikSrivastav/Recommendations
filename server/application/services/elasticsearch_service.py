from elasticsearch import Elasticsearch
import pandas as pd
from typing import List, Dict, Any, Optional
import ssl
import time

class ElasticsearchService:
    def __init__(self, host: str = "localhost", port: int = 9200, max_retries: int = 3):
        # Configure for Elasticsearch 8.x with authentication
        self.es = Elasticsearch(
            f"https://{host}:{port}",
            basic_auth=("elastic", "*jwy7uAOcSJ85IxXuICj"),
            verify_certs=False,  # In production, this should be True with proper cert verification
            ssl_show_warn=False,
            request_timeout=30,
            max_retries=max_retries,
            retry_on_timeout=True
        )
        self.index_name = 'songs'

    def create_index(self):
        """Create the songs index with appropriate mappings"""
        settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "song_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "asciifolding",
                                "word_delimiter_graph"
                            ]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "id": {"type": "integer"},
                    "track_title": {
                        "type": "text",
                        "analyzer": "song_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "completion": {
                                "type": "completion",
                                "analyzer": "song_analyzer"
                            }
                        }
                    },
                    "artist_name": {
                        "type": "text",
                        "analyzer": "song_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "completion": {
                                "type": "completion",
                                "analyzer": "song_analyzer"
                            }
                        }
                    },
                    "album_title": {
                        "type": "text",
                        "analyzer": "song_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "completion": {
                                "type": "completion",
                                "analyzer": "song_analyzer"
                            }
                        }
                    },
                    "track_genres": {"type": "keyword"},
                    "track_duration": {"type": "integer"},
                    "track_date_created": {
                        "type": "date",
                        "format": "strict_date_optional_time||epoch_millis"
                    },
                    "tags": {"type": "keyword"}
                }
            }
        }

        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name, body=settings)
            print(f"Created index '{self.index_name}' successfully")

    def index_songs(self, songs_df: pd.DataFrame):
        """Index songs from DataFrame into Elasticsearch"""
        print(f"Indexing {len(songs_df)} songs...")
        
        # Prepare bulk indexing data
        bulk_data = []
        for idx, song in songs_df.iterrows():
            # Index action
            bulk_data.append({"index": {"_index": self.index_name, "_id": idx}})
            
            # Document
            bulk_data.append({
                'id': int(idx),
                'track_title': song['track_title'],
                'artist_name': song['artist_name'],
                'album_title': song['album_title'],
                'track_genres': song.get('track_genres', []),
                'track_duration': song.get('track_duration'),
                'track_date_created': song.get('track_date_created'),
                'tags': song.get('tags', [])
            })

        # Perform bulk indexing
        if bulk_data:
            response = self.es.bulk(operations=bulk_data, refresh=True)
            if response.get('errors'):
                print("Some errors occurred during indexing")
            else:
                print(f"Successfully indexed {len(songs_df)} songs")

    def search_songs(self, query: str, limit: int = 5):
        """
        Search for songs using a multi-match query across multiple fields
        with different boosts and fuzziness
        """
        body = {
            "size": limit,
            "query": {
                "bool": {
                    "should": [
                        # Exact phrase matches (highest boost)
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "track_title^4",
                                    "artist_name^3",
                                    "album_title^2"
                                ],
                                "type": "phrase",
                                "boost": 4
                            }
                        },
                        # Fuzzy matches (medium boost)
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "track_title^3",
                                    "artist_name^2",
                                    "album_title"
                                ],
                                "fuzziness": "AUTO",
                                "boost": 2
                            }
                        },
                        # Prefix matches (lowest boost)
                        {
                            "bool": {
                                "should": [
                                    {"prefix": {"track_title": {"value": query, "boost": 1}}},
                                    {"prefix": {"artist_name": {"value": query, "boost": 1}}},
                                    {"prefix": {"album_title": {"value": query, "boost": 1}}}
                                ]
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "highlight": {
                "fields": {
                    "track_title": {},
                    "artist_name": {},
                    "album_title": {}
                },
                "pre_tags": ["<em>"],
                "post_tags": ["</em>"]
            }
        }

        try:
            response = self.es.search(index=self.index_name, body=body)
            total_hits = response['hits']['total']['value']
            hits = response['hits']['hits']

            results = []
            for hit in hits:
                source = hit['_source']
                results.append({
                    'id': source['id'],
                    'track_title': source['track_title'],
                    'artist_name': source['artist_name'],
                    'album_title': source['album_title'],
                    'track_genres': source.get('track_genres', []),
                    'track_duration': source.get('track_duration'),
                    'track_date_created': source.get('track_date_created'),
                    'tags': source.get('tags', []),
                    'score': hit['_score'],
                    'highlights': hit.get('highlight', {})
                })

            return results, total_hits
            
        except Exception as e:
            print(f"Search error: {str(e)}")
            return [], 0

    def get_suggestions(self, prefix: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get search suggestions as you type"""
        body = {
            "size": limit,
            "_source": ["track_title", "artist_name", "album_title"],
            "suggest": {
                "song_suggest": {
                    "prefix": prefix,
                    "completion": {
                        "field": "track_title.completion",
                        "size": limit,
                        "skip_duplicates": True
                    }
                },
                "artist_suggest": {
                    "prefix": prefix,
                    "completion": {
                        "field": "artist_name.completion",
                        "size": limit,
                        "skip_duplicates": True
                    }
                }
            }
        }

        try:
            response = self.es.search(index=self.index_name, body=body)
            suggestions = []
            
            # Process song suggestions
            for suggestion in response['suggest']['song_suggest'][0]['options']:
                suggestions.append({
                    'type': 'song',
                    'text': suggestion['_source']['track_title'],
                    'score': suggestion['_score']
                })
                
            # Process artist suggestions
            for suggestion in response['suggest']['artist_suggest'][0]['options']:
                suggestions.append({
                    'type': 'artist',
                    'text': suggestion['_source']['artist_name'],
                    'score': suggestion['_score']
                })
                
            # Sort by score and limit
            suggestions.sort(key=lambda x: x['score'], reverse=True)
            return suggestions[:limit]
            
        except Exception as e:
            print(f"Suggestion error: {str(e)}")
            return []

    def initialize_elasticsearch(self, songs_df: pd.DataFrame) -> bool:
        """
        Initialize Elasticsearch with songs data
        Returns True if successful, False otherwise
        """
        try:
            # Check connection
            if not self.es.ping():
                print("Could not connect to Elasticsearch")
                return False

            # Create index (will not recreate if exists)
            self.create_index()
            
            # Check if index is empty
            count = self.es.count(index=self.index_name)
            if count['count'] == 0:
                print("Index is empty, indexing songs...")
                self.index_songs(songs_df)
            else:
                print(f"Index already contains {count['count']} songs")
            
            return True
            
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            return False 