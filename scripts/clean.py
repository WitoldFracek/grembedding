import importlib
import os
import sys

from loguru import logger

if "DVC_ROOT" in os.environ.keys():
    root_dir = os.environ["DVC_ROOT"]
    if root_dir not in sys.path:
        sys.path.append(root_dir)
        logger.debug(f"Appending root dir: '{root_dir}' to sys.path")

from stages.datacleaners.DataCleaner import DataCleaner


def main():
    dataset: str = sys.argv[1]
    datacleaner: str = sys.argv[2]
    logger.info(f"Executing clean with dataset: '{dataset}' and datacleaner: '{datacleaner}'")

    datacleaner_cls = getattr(importlib.import_module(f"stages.datacleaners.{datacleaner}"), datacleaner)
    cleaner: DataCleaner = datacleaner_cls()
    logger.info(f"Instantiated datacleaner: '{cleaner.__class__.__name__}'")

    cleaner.clean_data(dataset)


if __name__ == "__main__":
    main()
