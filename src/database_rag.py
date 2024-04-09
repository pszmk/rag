r"""
This module constains functions for creating and managing the database.
"""

from pathlib import Path
from dotenv import load_dotenv, set_key
import pymongo
from typing import Tuple, List
import os

def _load_environment_variables() -> None:
    print("Loading environment variables...")
    load_dotenv()


def _get_db_client(
    credentials_var_name: str
    ) -> pymongo.MongoClient:
    r"""Returns the database client.
    """
    
    print("Getting database credentials...")
    try:
        db_client = os.getenv(credentials_var_name)
    except Exception as e:
        print(f"Error getting database credentials: {e}")
        return None
    
    print("Initializing database client...")
    try:
        client = pymongo.MongoClient(db_client)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None
    
    return client


def _insert_collection_into_db(
    client: pymongo.MongoClient,
    db_name: str,
    collection_name: str,
    data: List[dict]
    ) -> None:
    r"""Inserts a collection into the database.
    """
    
    assert isinstance(data, list), "Data must be a list of dictionaries."
    assert all(isinstance(d, dict) for d in data), "Data must be a list of dictionaries."
    
    print(f"Setting {db_name} database...")
    try:
        db = client[db_name]
    except Exception as e:
        print(f"Error setting database: {e}")
        return None
    
    print(f"Setting {collection_name} collection...")
    try:
        collection = db[collection_name]
    except Exception as e:
        print(f"Error setting collection: {e}")
        return None
    
    print("Inserting data into collection...")
    try:
        collection.insert_many(data)
    except Exception as e:
        print(f"Error inserting data into collection: {e}")
        return None


def _create_vector_search_index(
    collection: pymongo.collection.Collection,
    search_index_name: str,
    emebdding_field: str,
    embedding_dim: int
    ) -> None:
    r"""Creates a search index in the collection.
    """
    
    print(f"Creating {search_index_name} index...")
    definition = {
        "mappings": {
            "dynamic": True,
            "fields": {
                emebdding_field: {
                    "dimensions": embedding_dim,
                    "similarity": "cosine",
                    "type": "knnVector"
                }
            }
        }
        
    }
    try:
        collection.createSearchIndex(
            name=search_index_name,
            definition=definition
            )
    except Exception as e:
        print(f"Error creating index: {e}")
        return None
    
    print("Index created.")


def setup_database_and_collections(
    db_name: str,
    collections: List[Tuple[str, List[dict]]],
    # embedding_collection_number: int,
    # search_index_name: str,
    # emebdding_field: str,
    # embedding_dim: int
    ) -> None:
    r"""Sets up the medium collections in the database.
    """
    
    _load_environment_variables()
    client = _get_db_client('DB_CLIENT_ADMIN_CREDENTIALS')

    for collection in collections:
        collection_name, data = collection
        _insert_collection_into_db(client, db_name, collection_name, data)
    
    # embedding_colllection_name = collections[embedding_collection_number][0]
    # _create_vector_search_index(
    #     client[db_name][embedding_colllection_name],
    #     search_index_name,
    #     emebdding_field,
    #     embedding_dim
    #     )
    
    print("Database setup complete.")