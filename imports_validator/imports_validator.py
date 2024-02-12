import ast
import hashlib
import os
import time
from typing import List, Dict
import yaml

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
    return filename + ".yaml"


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

def _get_recursive_custom_imports(path: str) -> list[str]:
    imports = get_imports(path)
    imports = [imp.replace(".", "/") + ".py" for imp in imports]
    step_custom_imports = [imp for imp in imports if is_file(imp)]
    if not step_custom_imports:
        return []
    custom_imports = step_custom_imports.copy()
    for imp in step_custom_imports:
        custom_imports += _get_recursive_custom_imports(imp)
    return custom_imports


def validate(stage: str, stage_params: List[Dict[str, str]]):
    new_md5_df = pd.DataFrame(columns=['file', 'md5'])
    for run_params in stage_params:
        param_name = _get_important_param_name(stage)
        param_value = run_params[param_name]
        path = os.path.join("stages", f"{param_name}s", f"{param_value}.py")
        custom_imports = _get_recursive_custom_imports(path)
        md5s = []
        for imp in custom_imports:
            with open(imp, 'rb') as file:
                file_content = file.read()
                current_md5 = hashlib.md5(file_content).hexdigest()
                md5s.append(current_md5)

        md5s = sorted(md5s)
        path = os.path.join("imports_validator", stage)
        if not os.path.exists(path):
            os.mkdir(path)
        
        file_path = os.path.join(path, get_output_filename(stage, run_params))

        logger.info(f"Creating file {file_path}")
        with open(file_path, 'w') as file:
            yaml.dump(md5s, file)

    return new_md5_df

def main():
    with open('./params.yaml', 'r') as file:
        params = yaml.safe_load(file)

    validate("load", params['load'])
    validate("clean", params['clean'])
    validate("vectorize", params['vectorize'])
    validate("evaluate", params['classification_models'])
    validate("evaluate", params['clustering_models'])

if __name__ == "__main__":
    main()
