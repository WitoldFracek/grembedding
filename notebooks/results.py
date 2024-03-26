import pandas as pd
import json
import os
from pathlib import Path
from typing import Generator, Optional
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


def to_latex_table(
        df: pd.DataFrame, 
        place_modifiers: Optional[str] = None, 
        out_path: Optional[str | Path] = None, 
        border_style: str = '||', 
        separate_header: bool = False,
        column_names: Optional[list[str]] = None,
        float_precission: int = 2,
        caption: Optional[str] = None,
        label: Optional[str] = None,
        separate_rows: bool = False,
    ) -> str:
    """
    Function that changes data from DataFrame to LaTeX table.

    Args:
        df: DataFrame
        place_modifiers: Optional[str] - placing modifiers from LaTeX (for example 'h', 'ht', 'b').
        out_path: Optional[str | Path] - if given, saves the table to a file. Defaults to None.
        border_style: str - style of the border of the table. '|' single line border, '||' double line border. Defaults to '||'.
        separate_header: bool - whether the header should be visibly separated from the rest of the table. Defaults to False.
        column_names: Optional[list[str]] - if given this will indicate names in header. If set to None column names are the names of columns in DataFrame. Defaults to None.
        float_precission: int - rounds floats to the given precission. Defaults to 2.
        caption: Optional[str] - if given sets the caption of the generated table. Defaults to None.
        label: Optional[str] - if given sets the label of the generated table. Defaults to None.
        separate_rows: bool - whether separate the rows with a line. Defaults to False.
    """
    column_names = df.columns if column_names is None else column_names
    table = "\\begin{table}"
    if place_modifiers:
        table += f'[{place_modifiers}]'
    table += '\n\t\\centering\n\t\\begin{tabular}'
    table += f'{{{border_style}' + '|'.join(['c'] * len(df.columns)) + f'{border_style}}}\n\t\t\\hline\n\t\t'
    table += ' & '.join(column_names) + ' \\\\\n\t\t\\hline'
    if separate_header:
        table += '\\hline'
    table += '\n'
    for i, row in df.iterrows():
        data = ' & '.join(map(lambda s: s if isinstance(s, str) else f'{s:.{float_precission}f}' if isinstance(s, (int, float)) else str(s), row))
        table += '\t\t' + data + ' \\\\'
        if separate_rows:
            table += ' \\hline'
        table += '\n'
    if not separate_rows:
        table += '\t\t\\hline\n'
    table += '\t\\end{tabular}\n'
    if caption:
        table += f'\t\\caption{{{caption}}}\n'
    if label:
        table += f'\t\\label{{tab:{label}}}\n'
    table += '\\end{table}'
    if out_path:
        with open(out_path, 'w+', encoding='utf-8') as file:
            file.write(table)
    return table

