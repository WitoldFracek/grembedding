#!/usr/bin/env bash

dvc add datasets_raw  # add new data to dvc
dvc push

git add .
git commit -m "make dvc push" --allow-empty
git push