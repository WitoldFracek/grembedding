import os
import sys

from loguru import logger

if "DVC_ROOT" in os.environ.keys():
    root_dir = os.environ["DVC_ROOT"]
    sys.path.append(root_dir)
    logger.info(f"Appending root dir: '{root_dir}' to sys.path")

from stages.models.Model import Model
from stages.models.SVC import SVC

def main():
    dataset: str = sys.argv[1]
    datacleaner: str = sys.argv[2]
    vectorizer: str = sys.argv[3]
    model: str = sys.argv[4]

    md: Model = globals()[model]()
    md.evaluate(dataset, datacleaner, vectorizer)


if __name__ == "__main__":
    main()
