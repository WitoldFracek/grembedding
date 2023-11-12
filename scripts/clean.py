import os
import sys
import importlib

from loguru import logger

if "DVC_ROOT" in os.environ.keys():
    root_dir = os.environ["DVC_ROOT"]
    sys.path.append(root_dir)
    logger.info(f"Appending root dir: '{root_dir}' to sys.path")

from stages.datacleaners.DataCleaner import DataCleaner
from stages.datacleaners.LemmatizerSM import LemmatizerSM


def main():
    dataset: str = sys.argv[1]
    datacleaner: str = sys.argv[2]
    logger.info(f"Executing clean with dataset: '{dataset}' and datacleaner: '{datacleaner}'")

    dc: DataCleaner = globals()[datacleaner]()
    logger.info(f"Instantiated datacleaner: '{datacleaner}'")
    dc.clean_data(dataset)


if __name__ == "__main__":
    main()
