import os
import sys
from loguru import logger

if "DVC_ROOT" in os.environ.keys():
    root_dir = os.environ["DVC_ROOT"]
    sys.path.append(root_dir)
    logger.info(f"Appending root dir: '{root_dir}' to sys.path")

from stages.vectorizers.Vectorizer import Vectorizer
from stages.vectorizers.CountVectorizer import CountVectorizer


def main():

    dataset: str = sys.argv[1]
    datacleaner: str = sys.argv[2]
    vectorizer: str = sys.argv[3]

    v: Vectorizer = globals()[vectorizer]()
    v.vectorize(dataset, datacleaner)

if __name__ == "__main__":
    main()