import importlib
import os
import sys

from loguru import logger

if "DVC_ROOT" in os.environ.keys():
    root_dir = os.environ["DVC_ROOT"]
    if root_dir not in sys.path:
        sys.path.append(root_dir)
        logger.debug(f"Appending root dir: '{root_dir}' to sys.path")

from stages.vectorizers.Vectorizer import Vectorizer


def main():
    dataset_name: str = sys.argv[1]
    datacleaner_name: str = sys.argv[2]
    vectorizer_name: str = sys.argv[3]

    vectorizer_cls = getattr(importlib.import_module(f"stages.vectorizers.{vectorizer_name}"), vectorizer_name)
    v: Vectorizer = vectorizer_cls()
    logger.info(f"Instantiated vectorizer: '{v.__class__.__name__}'")

    v.vectorize(dataset_name, datacleaner_name)


if __name__ == "__main__":
    main()
