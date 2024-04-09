from typing import List, Optional
from src.indexing_rag import generate_query_embedding
from src.database_rag import vector_search
from src import utils_rag

def similarity_search(
    query: str,
    num_candidates: int,
    limit: int,
    ) -> Optional[List[dict]]:
    r"""Performs a similarity search.
    
    The function generates an embedding for the query and performs a similarity search in the specified search index.
    """
    
    query_embedding = generate_query_embedding(query)
    
    try:
        search_results_cursor = vector_search(
            query_embedding=query_embedding,
            num_candidates=num_candidates,
            limit=limit
            )
    except Exception as e:
        print(f"Error performing similarity search: {e}")
        return None
    
    search_results = {
        'id': [],
        'doc_id': [],
        'text_chunk': []
        }
    
    for result in search_results_cursor:
        id, doc_id, text_chunk, _ = tuple(result.values())
        search_results['id'].append(id)
        search_results['doc_id'].append(doc_id)
        search_results['text_chunk'].append(text_chunk)
    print("Search results end.")
    
    return search_results