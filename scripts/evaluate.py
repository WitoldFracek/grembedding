import importlib
import os
import sys

from loguru import logger

if "DVC_ROOT" in os.environ.keys():
    root_dir = os.environ["DVC_ROOT"]
    if root_dir not in sys.path:
        sys.path.append(root_dir)
        logger.debug(f"Appending root dir: '{root_dir}' to sys.path")

from stages.models.Model import Model


def main():
    dataset_name: str = sys.argv[1]
    datacleaner_name: str = sys.argv[2]
    vectorizer_name: str = sys.argv[3]
    model_name: str = sys.argv[4]

    model_cls = getattr(importlib.import_module(f"stages.models.{model_name}"), model_name)
    model: Model = model_cls()
    logger.info(f"Instantiated model: '{model.__class__.__name__}'")

    model.evaluate(dataset_name, datacleaner_name, vectorizer_name)


if __name__ == "__main__":
    main()
