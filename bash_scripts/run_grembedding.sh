#!/bin/bash

set -e

echo "Starting imports_validator..."
python ./imports_validator/imports_validator.py

if [ $? -eq 0 ]; then
    echo "imports_validator ended without error"
    echo "Starting dvc..."
    dvc exp run
else
    echo "Error during execution of the first command."
fi

PYTHONPATH=. python ./utils/mlflow/sync.py