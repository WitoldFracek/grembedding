#!/usr/bin/env bash

dvc add mlruns
dvc add datasets_raw
dvc push
git add .
git commit -m "make dvc push" --allow-empty
git push