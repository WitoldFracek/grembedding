<pre>
<code>
<p style="text-align: center;">
 _____                    _              _     _ _             
|  __ \                  | |            | |   | (_)            
| |  \/_ __ ___ _ __ ___ | |__   ___  __| | __| |_ _ __   __ _ 
| | __| '__/ _ \ '_ ` _ \| '_ \ / _ \/ _` |/ _` | | '_ \ / _` |
| |_\ \ | |  __/ | | | | | |_) |  __/ (_| | (_| | | | | | (_| |
 \____/_|  \___|_| |_| |_|_.__/ \___|\__,_|\__,_|_|_| |_|\__, |
                                                          __/ |
                                                         |___/
</p>
</code>
</pre>

# To run the pipeline
`bat` lub `sh`
```sh
bash_scripts/run_gremedding.bat
dvc push
git push
```

# To download pipeline
```sh
git pull
dvc pull
```

# To see mlflow results
```sh
mlflow ui
```

# Pipeline Parameters
All parameters found in config file: `params.yaml`

### load stage

- `dataloader`: name of the DataLoader class (base class `stages.dataloaders.DataLoader`)

### clean stage

- `dataloader`: name of the DataLoader class (base class `stages.dataloaders.DataLoader`)
- `datacleaner`: name of the DataCleaner class (base class `stages.datacleaners.DataCleaner`)

### vectorize stage:

- `dataloader`: name of the DataLoader class (base class `stages.dataloaders.DataLoader`)
- `datacleaner`: name of the DataCleaner class (base class `stages.datacleaners.DataCleaner`)
- `vectorizer`: name of the Vectorizer class (base class `stages.vectorizers.Vectorizer`)

### evaluate stage (classification and clustering):

- `dataloader`: name of the DataLoader class (base class `stages.dataloaders.DataLoader`)
- `datacleaner`: name of the DataCleaner class (base class `stages.datacleaners.DataCleaner`)
- `vectorizer`: name of the Vectorizer class (base class `stages.vectorizers.Vectorizer`)

### models parameters (classification and clustering):

- `model`: name of the Model class (base class `stages.models.Model`)
- `params`: name of the `.yaml` file with model parameters (found in `./params/`)


## Environment setup

1. Create a virtual environment with `python3 -m venv venv`. Supported versions: `3.10`, ...
2. Install dependencies using `pip install -r requirements.txt`.

## Requirements and versioning

### Use `venv` + `pip-tools` for dependency management.

1. Ensure you are using venv and have `pip-tools` installed.
2. Manage direct dependencies in `requirements.in`. You can pin the version to the newest one if you like.
Do not put transitive dependencies in `requirements.in` unless you want to pin them.
3. Use `pip-compile` to generate `requirements.txt`. This may take a bit to resolve all deps.
4. Use `pip-sync` to install the dependencies from `requirements.txt` into your virtual environment.

### Changing dependencies

1. Add / update / remove the dependency in `requirements.in`
2. Use `pip-compile && pip-sync` to sync venv with the new dependencies.
This is preferred to `pip install -r requirements.txt` because it will also uninstall unused dependencies.

### Install stylo_metrix
[Download pl_nask model](http://mozart.ipipan.waw.pl/~rtuora/spacy/)

```python
pip install stylo_metrix
pip install <pl_nask-0.0.7.tar.gz>
```

Pip may end with error. Incompatible between pl-core-news-* and spacy==3.5.4

```python
python -m spacy validate
python -m spacy download pl_core_news_sm
python -m spacy download pl_core_news_lg
```


### Spacy CUDA

1. Install MS Visual Studio (xD naprawdę potrzeba)
2. Install CUDA Toolkit (make note of version, I used latest - 12.x)
3. Remove Spacy `pip uninstall spacy`
4. Install Spacy with cuda (`cupy`) (get command from website [Spacy usage](https://spacy.io/usage)) - for CUDA 12.x it is `pip install -U 'spacy[cuda12x]==3.5.4'`
5. If `cupy-cuda` was not installed, install it manually `pip install cupy-cuda12x`
5. Test if cuda works with Spacy (notebook [](notebooks/spacy_cuda.ipynb))
6. If sth is f'd up you can try uninstalling `cupy-cuda12x` and installing it again

