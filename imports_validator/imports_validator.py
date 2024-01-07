import ast
import hashlib
import os
import time
from typing import List, Dict

import pandas as pd
import yaml
from loguru import logger


def get_imports(file_path):
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read(), filename=file_path)

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.add(f'{node.module}')
    return imports


def is_file(path):
    return os.path.exists(path)


def get_output_filename(stage: str, run_params: Dict[str, str]) -> str:
    filename = ''
    if stage == "load":
        filename = run_params['dataloader']
    elif stage == "clean":
        filename = f"{run_params['datacleaner']}"
    elif stage == "vectorize":
        filename = f"{run_params['vectorizer']}"
    elif stage == "evaluate":
        filename = f"{run_params['model']}"
    return filename + ".txt"


def _get_important_param_name(stage: str) -> str:
    if stage == "load":
        return "dataloader"
    if stage == "clean":
        return "datacleaner"
    if stage == "vectorize":
        return "vectorizer"
    if stage == "evaluate":
        return "model"
    raise ValueError(f"Stage {stage} does not exsist.")


def validate(stage: str, stage_params: List[Dict[str, str]], md5_df: pd.DataFrame):
    new_md5_df = pd.DataFrame(columns=['file', 'md5'])
    for run_params in stage_params:
        param_name = _get_important_param_name(stage)
        param_value = run_params[param_name]
        imports = get_imports(os.path.join("stages", f"{param_name}s", f"{param_value}.py"))
        imports = [imp.replace(".", "/") + ".py" for imp in imports]
        custom_imports = [imp for imp in imports if is_file(imp)]
        create_file_flag = False
        for imp in custom_imports:
            with open(imp, 'rb') as file:
                file_content = file.read()
                current_md5 = hashlib.md5(file_content).hexdigest()
                row = md5_df[md5_df['file'] == imp]
                if row.empty:
                    create_file_flag = True
                    new_md5_df = pd.concat([new_md5_df, pd.DataFrame({
                        "file": [imp],
                        "md5": [current_md5]
                    })], ignore_index=True)
                else:
                    last_md5 = row['md5'].values[0]
                    if last_md5 != current_md5:
                        create_file_flag = True
                        new_md5_df = pd.concat([new_md5_df, pd.DataFrame({
                            "file": [imp],
                            "md5": [current_md5]
                        })], ignore_index=True)

        if create_file_flag:
            path = os.path.join("imports_validator", stage)
            if not os.path.exists(path):
                os.mkdir(path)
            with open(os.path.join(path, get_output_filename(stage, run_params)), 'w') as file:
                logger.info(f"Creating file {os.path.join(path, get_output_filename(stage, run_params))}")
                file.write(str(time.time()))

    return new_md5_df


def main():
    with open('./params.yaml', 'r') as file:
        params = yaml.safe_load(file)
    if os.path.exists('./imports_validator/md5.parquet'):
        md5_df = pd.read_parquet('./imports_validator/md5.parquet')
    else:
        md5_df = pd.DataFrame(columns=['file', 'md5'])

    new_md5_dfs = [validate("load", params['load'], md5_df), validate("clean", params['clean'], md5_df),
                   validate("vectorize", params['vectorize'], md5_df), validate("evaluate", params['models'], md5_df)]

    new_md5_df = pd.concat(new_md5_dfs)
    new_md5_df = new_md5_df.drop_duplicates(subset='file')

    # Iteruj po każdym wierszu w md5_df
    for index, row in md5_df.iterrows():
        file_value = row['file']

        # Sprawdź, czy istnieje wiersz o tym samym pliku w new_md5_df
        if new_md5_df[new_md5_df['file'] == file_value].empty:
            # Jeśli nie istnieje, dodaj ten wiersz do new_md5_df
            new_row = {'file': file_value, 'md5': row['md5']}
            new_md5_df = pd.concat([new_md5_df, pd.DataFrame([new_row])], ignore_index=True)

    new_md5_df.to_parquet('./imports_validator/md5.parquet')


if __name__ == "__main__":
    main()
