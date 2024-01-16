import importlib
import os
import sys
import yaml

from loguru import logger

if "DVC_ROOT" in os.environ.keys():
    root_dir = os.environ["DVC_ROOT"]
    if root_dir not in sys.path:
        sys.path.append(root_dir)
        logger.debug(f"Appending root dir: '{root_dir}' to sys.path")

from stages.models.Model import Model
from utils.environment import get_root_dir


def main():
    dataset_name: str = sys.argv[1]
    datacleaner_name: str = sys.argv[2]
    vectorizer_name: str = sys.argv[3]
    tasks: str = sys.argv[4]
    model_name: str = sys.argv[5]
    params_name: str = sys.argv[6]
    task_name: str = sys.argv[7]

    with open(os.path.join(get_root_dir(), "params", f"{params_name}.yaml"), 'r') as file:
        params = yaml.safe_load(file)
    
    if task_name not  tasks.split(',')in:
        return

    model_cls = getattr(importlib.import_module(f"stages.models.{model_name}"), model_name)
    model: Model = model_cls()
    logger.info(f"Instantiated model: '{model.__class__.__name__}'")

    model.evaluate(dataset_name, datacleaner_name, vectorizer_name, params_name, params)


if __name__ == "__main__":
    main()
