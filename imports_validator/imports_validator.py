import os
import sys
import yaml
import ast
import time
from typing import List, Dict, Tuple


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

def get_output_mtime(stage: str, run_params: Dict[str, str]) -> float:
    output_file_path = ''
    if stage == "load":
        output_file_path = os.path.join("data", run_params['dataloader'], "raw")
    elif stage == "clean":
        output_file_path = os.path.join("data", run_params['dataloader'], run_params['datacleaner'])
    elif stage == "vectorize":
        output_file_path = os.path.join("data", run_params['dataloader'], f"{run_params['datacleaner']}_{run_params['vectorizer']}")
    elif stage == "evaluate":
        output_file_path = os.path.join("results", f"{run_params['model']}_{run_params['params']}")
    if os.path.exists(output_file_path):
        return os.path.getmtime(output_file_path)
    return -1

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

def validate(stage: str, stage_params: List[Dict[str, str]]):
    for run_params in stage_params:
        output_mtime = get_output_mtime(stage, run_params)
        for item in run_params.items():
            if item[0] == "params":
                imports = get_imports(os.path.join("params", f"{item[1]}.yaml"))
            else:
                imports = get_imports(os.path.join("stages", f"{item[0]}s", f"{item[1]}.py"))
            imports = [imp.replace(".", "/")+".py" for imp in imports]
            custom_imports = [imp for imp in imports if is_file(imp)]
            if any(os.path.getmtime(c_imp) > output_mtime for c_imp in custom_imports):
                path = os.path.join("imports_validator", stage)
                if not os.path.exists(path):
                    os.mkdir(path)
                with open(os.path.join(path, get_output_filename(stage, run_params)), 'w') as file:
                    file.write(str(time.time()))

def main():
    with open('./params.yaml', 'r') as file:
        params = yaml.safe_load(file)
        validate("load", params['load'])
        validate("clean", params['clean'])
        validate("vectorize", params['vectorize'])
        validate("evaluate", params['models'])


if __name__ == "__main__":
    main()