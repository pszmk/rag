# Project Name

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Description

A brief description of your project goes here.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Setup

To retrieve data use --retrieve flag and choose the correct name of the database with --db_name. The ```nokia_rag`` database is already setup for vector search and ready for querying. Please use this database name only for execution of the below command as the code runs with admin permissions.
```python main.py --retrieve --setup database_medium --db_name nokia_rag```

## Reproduction
Data preparation
To prepare the data for the document database
```python main.py --setup data --load_csv_source_filename "example.csv" --save_json_source_filename "example_org.json" --save_json_embeddings_filename "example_chunked_embeddings.json"```

The `"example.csv"` file is small just to show that the code runs. To run the code on the medium articles dataset set `"medium.csv"` as the source filename.

Note:
The the right filenams need to be used when launching the database.

To launch the database for the first time (done once per database). Inserting the data into the database.
```python main.py --launch --setup database_medium --db_name <unsearchable_demo_database_name> --medium_json_filename "example_org.json" --embeddings_json_filename "example_chunked_embeddings.json"```

Note:
Unfortunately for the free cluster tier I chose it is not possible to set search index in the database from code, so the database setup cannot  be fulle reproduced with just this repo. Some interaction with UI is necessary. For this reason for retrieval use already set up `nokia_rag` database.

## Usage

Examples and instructions on how to use your project.

## Contributing

Guidelines for contributing to the project.

## License

This project is licensed under the [MIT License](LICENSE).