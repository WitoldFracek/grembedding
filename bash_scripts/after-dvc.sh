#!/usr/bin/env bash

dvc add mlruns
dvc push
git add .
git commit -m "make dvc push" --allow-empty
git push