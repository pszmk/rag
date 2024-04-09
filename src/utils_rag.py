r"""
This module constains utility functions. 
"""

import pandas as pd
from pathlib import Path
import json
from typing import List, Tuple

DATA_PATH = Path(__file__).parent.parent / "data"

def load_data(
        data_load_filename: str
        ) -> Tuple[List, List[str]]:
    r"""A data loader.
    
    The data is loaded from the specified file.
    """
    
    available_extensions = [".csv", ".json"]
    assert any(data_load_filename.endswith(ext) for ext in available_extensions), f"data_load_filename must end with one of the following extensions: {available_extensions}"
    assert isinstance(data_load_filename, str), "data_filename must be a string"
    data_path = DATA_PATH / data_load_filename
    
    if data_load_filename.endswith('.csv'):
        data = pd.read_csv(data_path)
        assert 'Text' in data.columns, "data must contain a 'Text' column"
    
        return list(data.index), data['Title'].tolist(), data['Text'].tolist()
    
    elif data_load_filename.endswith('.json'):
        with open(data_path, 'r') as file_json:
            data = json.load(file_json)
        
        return data
    
    else:
        raise NotImplementedError(f"Unsupported file extension: {data_load_filename}")


def save_data_as_json_database_document(
    data: dict,
    data_save_filename: str
    ) -> None:
    r"""A data saving function.
    
    The data is saved as a JSON file as a set of documents.
    """
    
    assert '_id' == list(data.keys())[0], "The first key in data must be '_id'"
    assert all(isinstance(key, str) for key in data.keys()), "All keys in data must be strings"
    assert all(isinstance(value, list) for value in data.values()), "All values in data must be lists"
    assert data_save_filename.endswith('.json'), "data_save_filename must end with '.json'"
    
    data_path = DATA_PATH / data_save_filename
    
    data_like_json = [{key: value for key, value in zip(data.keys(), values)} for values in zip(*data.values())]

    with open(data_path, 'w') as file_json:
        json.dump(data_like_json, file_json, indent=2)
    
    return None
    
def save_output_to_txt(
    output: List[str],
    output_save_filename: str
    ) -> None:
    r"""A data saving function.
    
    The output is saved as a text file.
    """
    
    assert output_save_filename.endswith('.txt'), "output_save_filename must end with '.txt'"
    
    data_path = DATA_PATH / output_save_filename
    
    with open(data_path, 'w') as file_txt:
        for line in output:
            file_txt.write(line + '\n\n')
    
    return None