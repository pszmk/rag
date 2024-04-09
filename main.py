import argparse
from src import utils_rag
from src.indexing_rag import load_chunk_embed_save_as_json
from src.database_rag import setup_database_and_collections
import torch


def _build_parser():
    parser = argparse.ArgumentParser(description='RAG')
    parser.add_argument(
        '--setup',
        type=str,
        help='Setup the data or the database.',
        choices=['data', 'database_medium'],
        )
    
    data_group = parser.add_argument_group('data')
    data_group.add_argument(
        '--load_csv_source_filename',
        type=str,
        help='The name of the data file to load.',
        )
    data_group.add_argument(
        '--save_json_source_filename',
        type=str,
        help='The name of the data file to save.',
        )
    data_group.add_argument(
        '--save_json_embeddings_filename',
        type=str,
        help='The name of the embeddings file to save.',
        )
    data_group.add_argument(
        '--model_name',
        type=str,
        default='all-MiniLM-L6-v2',
        help='The name of the model to use.',
        )
    data_group.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='The device to use.',
        )
    data_group.add_argument(
        '--chunk_overlap',
        type=int,
        default=0,
        help='The overlap between chunks.',
        )
    data_group.add_argument(
        '--drop_last_token_number_threshold',
        type=int,
        default=None,
        help='The threshold for dropping the last chunk.',
        )
    
    database_medium_group = parser.add_argument_group('database_medium')
    database_medium_group.add_argument(
        '--db_name',
        type=str,
        help='The name of the database.',
        )
    database_medium_group.add_argument(
        '--medium_json_filename',
        type=str,
        help='The name of the medium json file.',
        )
    database_medium_group.add_argument(
        '--embeddings_json_filename',
        type=str,
        help='The name of the json file with the embeddings of the chunked medium articles.',
        )
    
    return parser


if __name__ == '__main__':
    parser = _build_parser()
    args = parser.parse_args()
    
    if args.setup == 'data':
        print(f"Loading data from {args.load_csv_source_filename} and saving to {args.save_json_source_filename}")
        ids, titles, texts = utils_rag.load_data(args.load_csv_source_filename)
        utils_rag.save_data_as_json_database_document(
            {
                '_id': ids,
                'Title': titles,
                'Text': texts
                },
            args.save_json_source_filename
            )
    
        load_chunk_embed_save_as_json(
            args.load_csv_source_filename,
            args.save_json_embeddings_filename,
            args.model_name,
            torch.device(args.device),
            args.chunk_overlap,
            args.drop_last_token_number_threshold
            )
    
    if args.setup == 'database_medium':
        
        medium_org_documents = utils_rag.load_data(
            args.medium_json_filename
            )
        
        medium_chunked_embeddings_documents = utils_rag.load_data(
            args.embeddings_json_filename,
            )
        
        collections = [
            ('medium_org', medium_org_documents),
            ('medium_chunked_embeddings', medium_chunked_embeddings_documents)
        ]
        
        setup_database_and_collections(
            args.db_name,
            collections
            )
        
        print('Warning!!! Setting up the vector search index needs to be done in the UI for the database cluster free tier I chose.')
            
