import pytest
import os
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from rag.src.indexing_rag import _load_emebedding_model, _load_text_splitter, _chunk_text, _generate_embeddings, load_chunk_embedd_save

DATA_PATH = Path(__file__).parent.parent / "data"

def test_load_emebedding_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "all-MiniLM-L6-v2"
    model = _load_emebedding_model(model_name, device)
    
    assert isinstance(model, SentenceTransformer)
    assert model.device == device
    
    model_name = "invalid_model_name"
    with pytest.raises(OSError):
        _load_emebedding_model(model_name, device)


def test_load_text_splitter():
    model_name = "all-MiniLM-L6-v2"
    splitter = _load_text_splitter(model_name)
    
    assert isinstance(splitter, SentenceTransformersTokenTextSplitter)
    
    model_name = "invalid_model_name"
    with pytest.raises(OSError):
        _load_text_splitter(model_name)


def test_chunk_text():
    texts = ["This is a test sentence.", "This is another test sentence."]
    splitter = _load_text_splitter("all-MiniLM-L6-v2")
    chunked = _chunk_text(texts, splitter)
    
    assert isinstance(chunked, list)
    assert len(chunked) == 2
    assert isinstance(chunked[0], list)
    
    maximum_tokens_per_chunk = splitter.maximum_tokens_per_chunk
    text = 'This is another test sentence.' * (maximum_tokens_per_chunk + 1)
    num_residual_tokens = splitter.count_tokens(text=text) % maximum_tokens_per_chunk
    num_chunks_when_drop_last = splitter.count_tokens(text=text) // maximum_tokens_per_chunk
    chunked = _chunk_text([text], splitter, num_residual_tokens)
    
    assert len(chunked[0]) == num_chunks_when_drop_last


def test_generate_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    text = ["This is a test sentence.", "This is another test sentence."]
    embeddings = _generate_embeddings(model, text)
    
    assert embeddings.shape == (2, 384)
    assert torch.is_tensor(embeddings)
    assert embeddings.device == model.device


def test_load_chunk_embedd_save():
    data_load_filename = "load_data_for_pytest_test.csv"
    data_save_filename = "save_data_for_pytest_test.pt"
    model_name = "all-MiniLM-L6-v2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_chunk_embedd_save(data_load_filename, data_save_filename, model_name, device)

    assert os.path.exists(DATA_PATH / data_save_filename)
    os.remove(DATA_PATH / data_save_filename)


