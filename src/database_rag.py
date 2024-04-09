r"""
This module constains functions for creating and managing the database.
"""

from pathlib import Path
from dotenv import load_dotenv, set_key
import pymongo
from typing import Tuple, List, Optional
import os

# This two are hardcoded as for now, because creating search index needs to be done in the UI. Could be stored in the config file though.
__SEARCH_INDEX_NAME = 'SemanticSearchIndex'
__EMBEDDING_FIELD = 'embedding'

__CLIENT_CREDS_VAR_NAME: str = 'DB_CLIENT_ADMIN_CREDENTIALS'

__DATABASE_NAME: Optional[str] = None
__ORG_COLLECTION_NAME: Optional[str] = None
__RETRIEVAL_COLLECTION_NAME: Optional[str] = None

__DB_CLIENT: Optional[pymongo.MongoClient] = None
__RETRIEVAL_COLLECTION: Optional[pymongo.collection.Collection] = None
__ORG_COLLECTION: Optional[pymongo.collection.Collection] = None
    

def _set_database_name(
    db_name: str
    ) -> None:
    r"""Sets the database name.
    """
    
    assert isinstance(db_name, str), "Database name must be a string."
    
    global __DATABASE_NAME
    if __DATABASE_NAME is not None:
        print("Database name already set.")
        return None
    else:
        __DATABASE_NAME = db_name
    
    return None


def _set_org_collection_name(
    collection_name: str,
    ) -> None:
    r"""Sets the org collection name.
    """
    
    assert isinstance(collection_name, str), "Collection name must be a string."
    
    global __ORG_COLLECTION_NAME
    if __ORG_COLLECTION_NAME is not None:
        print("Org collection name already set.")
        return None
    else:
        __ORG_COLLECTION_NAME = collection_name
        
    return None


def _set_retrieval_collection_name(
    collection_name: str,
    ) -> None:
    r"""Sets the retrieval collection name.
    """
    
    assert isinstance(collection_name, str), "Collection name must be a string."
    
    global __RETRIEVAL_COLLECTION_NAME
    if __RETRIEVAL_COLLECTION_NAME is not None:
        print("Retrieval collection name already set.")
        return None
    else:
        __RETRIEVAL_COLLECTION_NAME = collection_name
    
    return None


def _load_environment_variables() -> None:
    print("Loading environment variables...")
    load_dotenv()
    
    return None


def _set_db_client(
    credentials_var_name: str
    ) -> Optional[pymongo.MongoClient]:
    r"""Returns the database client.
    """
    
    global __DB_CLIENT
    if __DB_CLIENT is not None:
        print("Database client already set.")
        return None
    
    print("Getting database credentials...")
    try:
        db_client_credentials = os.getenv(credentials_var_name)
    except Exception as e:
        print(f"Error getting database credentials: {e}")
        return None
    
    print("Initializing database client...")
    try:
        db_client = pymongo.MongoClient(db_client_credentials)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None
    
    __DB_CLIENT = db_client


def _get_db_client() -> Optional[pymongo.MongoClient]:
    r"""Returns the database client.
    """
    
    global __DB_CLIENT
    if __DB_CLIENT is not None:
        return __DB_CLIENT
    else:
        raise ValueError("Database client not set.")


def _set_org_collection() -> None:
    r"""Sets the org collection.
    """
    
    global __ORG_COLLECTION
    if __ORG_COLLECTION is not None:
        print("Org collection already set.")
        return None
    
    if __ORG_COLLECTION_NAME is None:
        raise ValueError("Org collection name not set.")
    
    global __DB_CLIENT
    if __DB_CLIENT is None:
        raise ValueError("Database client not set.")
    
    print("Getting org collection...")
    try:
        org_collection = __DB_CLIENT[__DATABASE_NAME][__ORG_COLLECTION_NAME]
    except Exception as e:
        print(f"Error getting org collection: {e}")
        return None
    
    __ORG_COLLECTION = org_collection
    
    return None


def _get_org_collection() -> pymongo.collection.Collection:
    r"""Returns the org collection.
    """
    
    global __ORG_COLLECTION
    if __ORG_COLLECTION is not None:
        return __ORG_COLLECTION
    else:
        raise ValueError("Org collection not set.")


def _set_retrieval_collection() -> None:
    r"""Sets the retrieval collection.
    """
    
    global __RETRIEVAL_COLLECTION
    if __RETRIEVAL_COLLECTION is not None:
        print("Retrieval collection already set.")
        return None
    
    if __RETRIEVAL_COLLECTION_NAME is None:
        raise ValueError("Retrieval collection name not set.")
    
    global __DB_CLIENT
    if __DB_CLIENT is None:
        raise ValueError("Database client not set.")
    
    print("Getting retrieval collection...")
    try:
        retrieval_collection = __DB_CLIENT[__DATABASE_NAME][__RETRIEVAL_COLLECTION_NAME]
    except Exception as e:
        print(f"Error getting retrieval collection: {e}")
        return None
    
    __RETRIEVAL_COLLECTION = retrieval_collection
    
    return None


def _get_retrieval_collection() -> pymongo.collection.Collection:
    r"""Returns the retrieval collection.
    """
    
    global __RETRIEVAL_COLLECTION
    if __RETRIEVAL_COLLECTION is not None:
        return __RETRIEVAL_COLLECTION
    else:
        raise ValueError("Retrieval collection not set.")


def _insert_collection_into_db(
    collection: pymongo.collection.Collection,
    data: List[dict]
    ) -> None:
    r"""Inserts a collection into the database.
    """
    
    assert collection is not None, "Collection must not be None."
    assert isinstance(data, list), "Data must be a list of dictionaries."
    assert all(isinstance(d, dict) for d in data), "Data must be a list of dictionaries."
    
    print("Inserting data into collection...")
    try:
        collection.insert_many(data)
    except Exception as e:
        print(f"Error inserting data into collection: {e}")
        return None
    
    return None


def _create_vector_search_index(
    search_index_name: str,
    emebdding_field: str,
    embedding_dim: int
    ) -> None:
    r"""Creates a search index in the collection.
    """
    
    collection = _get_org_collection()
    
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
    org_collection_name: str,
    retrieval_collection_name: str
    ) -> None:
    r"""Sets up the medium collections in the database.
    """
    
    _set_database_name(db_name)
    _set_org_collection_name(org_collection_name)
    _set_retrieval_collection_name(retrieval_collection_name)
    
    _load_environment_variables()
    _set_db_client(__CLIENT_CREDS_VAR_NAME)
    
    _set_org_collection()
    _set_retrieval_collection()
    
    print("Database setup complete.")
    
    return None
    

def launch_database_and_collections(
    db_name: str,
    collections_data: List[Tuple[str, List[dict]]],
    embeddings_collection_number: int
    ) -> None:
    r"""Sets up the medium collections in the database.
    """
    
    assert isinstance(db_name, str), "Database name must be a string."
    assert isinstance(collections_data, list), "Collections must be a list of tuples."
    assert len(collections_data) == 2, "There must be two collections."
    assert embeddings_collection_number in [0, 1], "Embedding collection number must be 0 or 1."
    
    collections_data = collections_data if embeddings_collection_number == 1 else collections_data[::-1]
    org_collection_name, retrieval_collection_name = collections_data[0][0], collections_data[1][0]
    
    setup_database_and_collections(
        db_name,
        org_collection_name,
        retrieval_collection_name
        )
    
    org_data, retrieval_data = collections_data[0][1], collections_data[1][1]
    _insert_collection_into_db(_get_org_collection(), org_data)
    _insert_collection_into_db(_get_retrieval_collection(), retrieval_data)
    
    print("Collections inserted into database.")
    
    return None
    

def vector_search(
    query_embedding: List[float],
    num_candidates: int,
    limit: int,
    ) -> Optional[List[dict]]: 
    r"""Searches the collection for the most similar vectors to the query vector.
    """
    
    assert isinstance(query_embedding, list), "Query embedding must be a list of floats."
    assert all(isinstance(e, float) for e in query_embedding), "Query embedding must be a list of floats."
    assert isinstance(num_candidates, int) and num_candidates > 0, "Num candidates must be a positive integer."
    assert isinstance(limit, int) and limit > 0, "Limit must be a positive integer."
    
    collection = _get_retrieval_collection()
    
    try:
        results = collection.aggregate([
            {"$vectorSearch": {
                "queryVector": query_embedding,
                "path": __EMBEDDING_FIELD,
                "numCandidates": num_candidates,
                "limit": limit,
                "index": __SEARCH_INDEX_NAME,
                }
             }
            ])
    except Exception as e:
        print(f"Error searching collection: {e}")
        return None
    
    return results