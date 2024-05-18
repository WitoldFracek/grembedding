import pandas as pd
import json
import os
from pathlib import Path
from typing import Generator, Optional, Iterable, Callable
from copy import deepcopy
import pandas.api.typing as pdtype


class SupportsRichComparison:
    pass

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


class GremDataFrame(pd.DataFrame):
    def __init__(self, other: pd.DataFrame):
        super().__init__(other)
    
    def dataset(self, name: str | list[str]) -> "GremDataFrame":
        allowed = [name] if isinstance(name, str) else name
        return GremDataFrame(self[self['dataset'].isin(allowed)])

    def data_cleaner(self, name: str | list[str]) -> "GremDataFrame":
        allowed = [name] if isinstance(name, str) else name
        return GremDataFrame(self[self['datacleaner'].isin(allowed)])
    
    def vectorizer(self, name: str | list[str]) -> "GremDataFrame":
        allowed = [name] if isinstance(name, str) else name
        return GremDataFrame(self[self['vectorizer'].isin(allowed)])
    
    def classification(self) -> "GremDataFrame":
        return GremDataFrame(self[~self['f1_score'].isna()])
    
    def clusterization(self) -> "GremDataFrame":
        return GremDataFrame(self[self['f1_score'].isna()])
    
    def index_to_col(self, column_name: str) -> "GremDataFrame":
        df = self.copy()
        df[column_name] = df.index
        df = df[[column_name] + list(df.columns)[:-1]]
        return GremDataFrame(df)
    
    def first(self) -> "GremDataFrame":
        return GremDataFrame(super().first())


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


def include_pivot_index(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    df[column_name] = df.index
    df = df[[column_name] + list(df.columns)[:-1]]
    return df

def make_latex_frame(
        columns: list[str],
        place_modifiers: Optional[str] = None,
        border_style: str = '||',
        separate_header: bool = False,
        caption: Optional[str] = None,
        column_names: Optional[list[str]] = None,
        bold_labels: Optional[list[tuple[str, str] | str]] = None,
        label: Optional[str] = None,
        separate_rows: bool = False,
    ) -> tuple[str, str]:
    column_names = columns if column_names is None else column_names
    bold_labels = [] if bold_labels is None else bold_labels    

    begin_table = "\\begin{table}"
    if place_modifiers:
        begin_table += f'[{place_modifiers}]'
    begin_table += '\n\t\\centering\n\t\\caption{'
    if caption:
        begin_table += caption
    begin_table += '}\n\t\\resizebox{\\textwidth}{!}{\n\t\\begin{tabular}'
    begin_table += f'{{{border_style}' + '|'.join(['c'] * len(columns)) + f'{border_style}}}\n\t\t\\hline\n\t\t'
    begin_table += ' & '.join(map(lambda s: s.replace("_", "\\_"), map(lambda s: f"\\textbf{{{s}}}", column_names))) + ' \\\\\n\t\t\\hline'
    if separate_header:
        begin_table += '\\hline'
    begin_table += '\n'

    end_table = ''
    if not separate_rows:
        end_table += '\t\t\\hline\n'
    end_table += '\t\\end{tabular}\n'
    end_table += '\t}\n'
    end_table += '\t\\label{tab:'
    if label:
        end_table += f'{label}'
    end_table += '}\n'
    end_table += '\\end{table}'
    return begin_table, end_table


def to_latex_table(
        df: pd.DataFrame, 
        place_modifiers: Optional[str] = 'H', 
        out_path: Optional[str | Path] = None, 
        border_style: str = '||', 
        separate_header: bool = False,
        column_names: Optional[list[str]] = None,
        float_precission: int = 3,
        caption: Optional[str] = None,
        label: Optional[str] = None,
        separate_rows: bool = False,
        bold_labels: Optional[list[tuple[str, str] | str]] = None
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
    bold_labels = [] if bold_labels is None else bold_labels
    begin_table, end_table = make_latex_frame(
        columns=df.columns,
        place_modifiers=place_modifiers,
        border_style=border_style,
        separate_header=separate_header,
        separate_rows=separate_rows,
        caption=caption,
        label=label,
        bold_labels=bold_labels,
        column_names=column_names
    )

    max_scores = {}
    for elem in bold_labels:
        if isinstance(elem, str):
            score_label = elem
            mode = 'max'
        elif isinstance(elem, tuple):
            score_label, mode = elem
        else:
            raise Exception('bold_labels item is not str nor tuple[str, str]')
        max_scores[score_label] = f'{df[score_label].max():.{float_precission}f}' if mode == 'max' else f'{df[score_label].min():.{float_precission}f}'
    
    table = ''
    for _, row in df.iterrows():
        data = __generate_table_row(row, df.columns, max_scores, float_precission)
        table += '\t\t' + data + ' \\\\'
        if separate_rows:
            table += ' \\hline'
        table += '\n'
    
    table = begin_table + table + end_table
    if out_path:
        with open(out_path, 'w+', encoding='utf-8') as file:
            file.write(table)
    return table


def __generate_table_row(data_row: Iterable, data_labels: list[str], bold_mappings: dict[str, str], float_precission) -> str:
    strs = []
    for data, label in zip(data_row, data_labels):
        data = data if isinstance(data, str) else f'{data:.{float_precission}f}' if isinstance(data, (int, float)) else str(data)
        data = data.replace('_', '\\_')
        if label in bold_mappings:
            if data == bold_mappings[label]:
                data = f'\\textbf{{{data}}}'
        strs.append(data)
    return ' & '.join(strs)


def __generate_group(group: pd.DataFrame, group_name: str, data_labels: list[str], bold_mappings: dict[str, str], float_precission: int, separate_rows: bool) -> str:
    big_cell = f'\\multirow{{{len(group)}}}{{*}}{{{group_name}}} & '
    table = ''
    for i, (_, row) in enumerate(group.iterrows()):
        first_cell = big_cell if i == 0 else ' & '
        data = first_cell + __generate_table_row(row, data_labels, bold_mappings, float_precission)
        table += '\t\t' + data + ' \\\\'
        if separate_rows:
            table += f'  \\cline{{2-{len(data_labels) + 1}}}'
        table += '\n'
    table += f'\\hline'
    return table


def groups_to_latex_table(
        gdf: pdtype.DataFrameGroupBy,
        group_header: str = 'rodzaj wektora' ,
        place_modifiers: Optional[str] = 'H', 
        out_path: Optional[str | Path] = None, 
        border_style: str = '||', 
        separate_header: bool = False,
        column_names: Optional[list[str]] = None,
        float_precission: int = 3,
        caption: Optional[str] = None,
        label: Optional[str] = None,
        separate_rows: bool = False,
        bold_labels: Optional[list[tuple[str, str] | str]] = None,
        groups_sort_key: Optional[Callable[[tuple[str, pd.DataFrame]], SupportsRichComparison]] = None
    ) -> str:
    bold_labels = [] if bold_labels is None else bold_labels
    _, df = next(iter(gdf))
    begin_table, end_table = make_latex_frame(
        columns=[group_header] + list(df.columns),
        place_modifiers=place_modifiers,
        border_style=border_style,
        separate_header=separate_header,
        separate_rows=separate_rows,
        caption=caption,
        label=label,
        bold_labels=bold_labels,
        column_names=column_names
    )
    max_scores = {}
    for group_name, group in gdf:
        max_scores[group_name] = {}
        for elem in bold_labels:
            if isinstance(elem, str):
                score_label = elem
                mode = 'max'
            elif isinstance(elem, tuple):
                score_label, mode = elem
            else:
                raise Exception('bold_labels item is not str nor tuple[str, str]')
            max_scores[group_name][score_label] = f'{group[score_label].max():.{float_precission}f}' if mode == 'max' else f'{group[score_label].min():.{float_precission}f}'
    table = ''
    groups = list(iter(gdf))
    if groups_sort_key is not None:
        groups = sorted(groups, key=groups_sort_key)
    for group_name, group in groups:
        group_str = __generate_group(group, group_name, group.columns, max_scores[group_name], float_precission, separate_rows)
        table += group_str
    table = table[:-6]  # remove doubled \hline

    table = begin_table + table + end_table
    if out_path:
        with open(out_path, 'w+', encoding='utf-8') as file:
            file.write(table)
    return table