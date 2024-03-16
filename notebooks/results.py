import pandas as pd
import json
import os
from pathlib import Path
from typing import Generator
from copy import deepcopy

RESULTS_DIR = Path(os.path.join('..', 'results'))
METRICS = [
    'accuracy', 
    'f1_score', 
    'recall', 
    'precision', 
    'silhouette', 
    'davies_bouldin',
    'calinski_harabasz', 
    'bcubed_precission', 
    'bcubed_recall', 
    'bcubed_f1'
]

__DATA_DICT = {
    'dataset': [],
    'datacleaner': [],
    'vectorizer': [],
    'params_name': [],
}
for metric in METRICS:
    __DATA_DICT[metric] = []


def results_iter(root_dir: str | Path) -> Generator[dict[str, dict | str | float], None, None]:
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as file:
                    yield json.load(file)
        for dir_ in dirs:
            results_iter(os.path.join(root, dir_))


def __update_data(results, data: dict[str, list[str | float]]):
    datacleaner = results['datacleaner']
    dataset = results['dataset']
    vectorizer = results['vectorizer']
    params_name = results['params_name']
    met: dict[str, float] = results['metrics']

    data['datacleaner'].append(datacleaner)
    data['dataset'].append(dataset)
    data['vectorizer'].append(vectorizer)
    data['params_name'].append(params_name)
    for metric in METRICS:
        data[metric].append(met.get(metric, None))


def load_results(results_dir: str | Path) -> pd.DataFrame:
    data = deepcopy(__DATA_DICT)
    for results in results_iter(results_dir):
        __update_data(results, data)
    results_df = pd.DataFrame.from_dict(data)
    return results_df


def classification(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df['bcubed_f1'].isna()]


def clusterization(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df['accuracy'].isna()]
