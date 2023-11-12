import sys

from loguru import logger

from objects.datacleaners.DataCleaner import DataCleaner
from objects.datacleaners.LemmatizerSM import LemmatizerSM


def main():
    dataset: str = sys.argv[1]
    datacleaner: str = sys.argv[2]
    logger.info(f"Executing clean with dataset: '{dataset}' and datacleaner: '{datacleaner}'")

    dc: DataCleaner = globals()[datacleaner]()
    logger.info(f"Instantiated datacleaner: '{datacleaner}'")
    dc.clean_data(dataset)


if __name__ == "__main__":
    main()
