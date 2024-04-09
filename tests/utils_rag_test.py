import pytest
from pathlib import Path
from rag.src import utils_rag

DATA_PATH = Path(__file__).parent.parent / "data"

def test_load_data():
    data_load_filename = "load_data_for_pytest_test.csv"
    data = utils_rag.load_data(data_load_filename, "csvpandas")

    assert isinstance(data, tuple)
    assert isinstance(data[0], list)
    assert isinstance(data[1], list)
    assert len(data[0]) == 2
    assert len(data[1]) == 2
    assert data[1][0] == "This is a test sentence."
    assert data[1][1] == "This is another test sentence."


def test_save_data_as_json_database_document():
    data = {
        "_id": ["1", "2"],
        "Title": ["Test 1", "Test 2"],
        "Text": ["This is a test sentence.", "This is another test sentence."]
    }
    data_save_filename = "save_data_as_json_database_doc_for_pytest_test.json"
    utils_rag.save_data_as_json_database_doc(data, data_save_filename)

    data_path = DATA_PATH / data_save_filename
    assert data_path.exists()

    with open(data_path, 'r') as f:
        data_like_json = json.load(f)
    
    assert isinstance(data_like_json, list)
    assert len(data_like_json) == 2
    assert data_like_json[0] == {"_id": "1", "Title": "Test 1", "Text": "This is a test sentence."}
    assert data_like_json[1] == {"_id": "2", "Title": "Test 2", "Text": "This is another test sentence."}

    data_path.unlink()
    assert not data_path.exists()