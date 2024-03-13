#!/bin/bash

set -e

echo "Starting imports_validator..."
python ./imports_validator/imports_validator.py

if [ $? -eq 0 ]; then
    echo "imports_validator ended without error"
    echo "Starting dvc..."

    # Env var do sterowania spacy
    export GRE_SPACY_MODE=gpu
    export GRE_SPACY_BATCH_SIZE=64

    dvc exp run
else
    echo "Error during execution of the first command."
fi

PYTHONPATH=. python ./utils/mlflow/sync.py