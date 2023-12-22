import os
import sys
import yaml
import ast
import time
from typing import List, Dict, Tuple
from loguru import logger
import pandas as pd
import hashlib


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

# def get_output_mtime(stage: str, run_params: Dict[str, str]) -> float:
#     output_file_path = ''
#     if stage == "load":
#         output_file_path = os.path.join("data", run_params['dataloader'], "raw")
#     elif stage == "clean":
#         output_file_path = os.path.join("data", run_params['dataloader'], run_params['datacleaner'])
#     elif stage == "vectorize":
#         output_file_path = os.path.join("data", run_params['dataloader'], f"{run_params['datacleaner']}_{run_params['vectorizer']}")
#     elif stage == "evaluate":
#         output_file_path = os.path.join("results", f"{run_params['model']}_{run_params['params']}")
#     if os.path.exists(output_file_path):
#         return os.path.getmtime(output_file_path)
#     return -1

def get_output_filename(stage: str, run_params: Dict[str, str]) -> str:
    filename = ''
    if stage == "load":
        filename = run_params['dataloader']
    elif stage == "clean":
        filename = f"{run_params['dataloader']}_{run_params['datacleaner']}"
    elif stage == "vectorize":
        filename = f"{run_params['dataloader']}_{run_params['datacleaner']}_{run_params['vectorizer']}"
    elif stage == "evaluate":
        filename = f"{run_params['model']}_{run_params['params']}"
    return filename+".txt"

def validate(stage: str, stage_params: List[Dict[str, str]], md5_df: pd.DataFrame):
    for run_params in stage_params:
        # output_mtime = get_output_mtime(stage, run_params)
        for item in run_params.items():
            if item[0] == "params":
                imports = get_imports(os.path.join("params", f"{item[1]}.yaml"))
            else:
                imports = get_imports(os.path.join("stages", f"{item[0]}s", f"{item[1]}.py"))
            imports = [imp.replace(".", "/")+".py" for imp in imports]
            custom_imports = [imp for imp in imports if is_file(imp)]
            create_file_flag = False
            for imp in custom_imports:
                with open(imp, 'rb') as file:
                    file_content = file.read()
                    current_md5 = hashlib.md5(file_content).hexdigest()
                    row = md5_df[md5_df['file'] == imp]
                    if row.empty:
                        create_file_flag = True
                        md5_df = pd.concat([md5_df, pd.DataFrame({
                            "file": [imp],
                            "md5": [current_md5]
                        })], ignore_index=True)
                    else:
                        last_md5 = row['md5'].values[0]
                        if last_md5 != current_md5:
                            create_file_flag = True
                            md5_df.loc[md5_df['file'] == imp, 'md5'] = current_md5
            
            if create_file_flag:
                path = os.path.join("imports_validator", stage)
                if not os.path.exists(path):
                    os.mkdir(path)
                with open(os.path.join(path, get_output_filename(stage, run_params)), 'w') as file:
                    logger.info(f"Creating file {os.path.join(path, get_output_filename(stage, run_params))}")
                    file.write(str(time.time()))
                    
                    

            # if any(os.path.getmtime(c_imp) > output_mtime for c_imp in custom_imports):
            #     path = os.path.join("imports_validator", stage)
            #     if not os.path.exists(path):
            #         os.mkdir(path)
            #     with open(os.path.join(path, get_output_filename(stage, run_params)), 'w') as file:
            #         logger.info(f"Creating directory {os.path.join(path, get_output_filename(stage, run_params))}")
            #         file.write(str(time.time()))
    return md5_df

def main():
    with open('./params.yaml', 'r') as file:
        params = yaml.safe_load(file)
    if os.path.exists('./imports_validator/md5.pickle'):
        md5_df = pd.read_parquet('./imports_validator/md5.pickle')
    else:
        md5_df = pd.DataFrame(columns = ['file', 'md5'])
    md5_df = validate("load", params['load'], md5_df)
    md5_df = validate("clean", params['clean'], md5_df)
    md5_df = validate("vectorize", params['vectorize'], md5_df)
    md5_df = validate("evaluate", params['models'], md5_df)
    md5_df.to_parquet('./imports_validator/md5.pickle')


if __name__ == "__main__":
    main()