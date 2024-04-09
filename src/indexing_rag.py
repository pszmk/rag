r"""
This module contains functions for creating vector embedings and preparing the data for indexing.
Embeddings are generated using the SentenceTransformer library.
"""

from pathlib import Path
from typing import Optional, List, Tuple, Union
from numpy import ndarray
import pandas as pd
import torch
from sentence_transformers import util, SentenceTransformer
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from tqdm import tqdm

from src import utils_rag

DATA_PATH = Path('data') #DATA_PATH = Path(__file__).parent.parent / "data"

def _load_emebedding_model(
    model_name: str,
    device: torch.device
    ) -> SentenceTransformer:
    r"""A model loader.

    The specified model is loaded using the SentenceTransformer library on the specified device.
    """

    assert isinstance(device, torch.device), "device must be a torch.device object"

    try:
        model = SentenceTransformer(
            model_name_or_path=model_name,
            device=device
        )
    except OSError as e:
        raise OSError(f"Error loading model: {e}")

    return model


def _load_text_splitter(
        model_name: str,
        chunk_overlap: int = 0
        ) -> SentenceTransformersTokenTextSplitter:
    r"""A text splitter loader.

    The specified model is loaded using the SentenceTransformersTokenTextSplitter class.
    """
    assert isinstance(model_name, str), "model_name must be a string"
    assert isinstance(chunk_overlap, int) and chunk_overlap >= 0, "chunk_overlap must be a non-negative integer"

    try:
        splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap,
            model_name=model_name
        )
    except OSError as e:
        raise OSError(f"Error loading text splitter: {e}")

    return splitter


def _generate_embeddings(
        model: SentenceTransformer,
        texts: List[str],
        output_mode: Optional[str] = None
        ) -> Union[ndarray, torch.Tensor, List[List[float]]]:
    r"""An embedding generator.
    
    The list of texts is converted to embeddings using the specified model.
    """
    
    assert isinstance(model, SentenceTransformer), "model must be a SentenceTransformer object"
    assert isinstance(texts, list), "texts must be a list of strings"
    available_output_modes = [None, 'ndarray', 'list', 'tensor_default', 'tensor_model_device']
    assert output_mode is None or output_mode in available_output_modes, f"output_mode must be in {available_output_modes}"
    
    if output_mode is None or output_mode == 'ndarray':
        embeddings = model.encode(texts, convert_to_tensor=False)
    elif output_mode == 'list':
        embeddings = model.encode(texts, convert_to_tensor=False).tolist()
    elif output_mode == 'tensor_default':
        embeddings = model.encode(texts, convert_to_tensor=True)
    elif output_mode == 'tensor_model_device':
        embeddings = model.encode(texts, convert_to_tensor=True).to(model.device)

    return embeddings


def _chunk_text(
        texts: List[str],
        splitter: SentenceTransformersTokenTextSplitter,
        drop_last_token_number_threshold: Optional[int] = None
        ) -> List[List[str]]:
    r"""A text chunker.

    The text is split into chunks of the specified token length.
    The last chunk is dropped if it contains fewer tokens than the specified threshold.
    """

    assert isinstance(splitter, SentenceTransformersTokenTextSplitter), "splitter must be a SentenceTransformersTokenTextSplitter object"
    assert isinstance(texts, list), "text must be a list of strings"
    assert drop_last_token_number_threshold is None or (isinstance(drop_last_token_number_threshold, int) and drop_last_token_number_threshold >= 0 and drop_last_token_number_threshold <= splitter.maximum_tokens_per_chunk), "drop_last_token_number_threshold must be None or be an integer between 0 and the maximum number of tokens per chunk"

    chunked = []
    for text in texts:
        assert isinstance(text, str), "text must be a list of strings"
        
        chunks = splitter.split_text(text)
        
        if drop_last_token_number_threshold and len(chunks) > 1 and splitter.count_tokens(text=chunks[-1]) <= drop_last_token_number_threshold:
            chunks.pop()
        
        chunked.append(chunks)
    
    return chunked


def load_chunk_embed_save_as_json(
        data_load_filename: str,
        data_save_filename: str,
        model_name: str,
        device: torch.device,
        chunk_overlap: int = 0,
        drop_last_token_number_threshold: Optional[int] = None
        ) -> None:
    r"""A data loader.
    
    If batch provided, the data is loaded in batches.
    """

    assert isinstance(data_load_filename, str), "data_filename must be a string"
    assert isinstance(data_save_filename, str), "data_save_filename must be a string"

    indices, _, texts = utils_rag.load_data(data_load_filename=data_load_filename)

    model = _load_emebedding_model(model_name, device)
    splitter = _load_text_splitter(model_name, chunk_overlap)
    chunked = _chunk_text(texts, splitter, drop_last_token_number_threshold)

    document_ids = []
    chunked_flat = []
    chunked_embeddings = []
    
    progress_bar = tqdm(zip(indices, chunked), total=len(chunked), desc="Generating embeddings")
    for document_id, chunks in progress_bar:
        chunks_embeddings = _generate_embeddings(model, chunks, 'list')
        for chunk, chunk_embedding in zip(chunks, chunks_embeddings):
            document_ids.append(document_id)
            chunked_flat.append(chunk)
            chunked_embeddings.append(chunk_embedding)
        
    chunk_ids = list(range(len(chunked_flat)))
    utils_rag.save_data_as_json_database_document(
        {'_id': chunk_ids, 'document_id': document_ids, 'text_chunk': chunked_flat, 'embedding': chunked_embeddings},
        data_save_filename
        )
    
    return None