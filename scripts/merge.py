import importlib
import os
import sys
import numpy as np
from utils.environment import get_root_dir

from loguru import logger

if "DVC_ROOT" in os.environ.keys():
    root_dir = os.environ["DVC_ROOT"]
    if root_dir not in sys.path:
        sys.path.append(root_dir)
        logger.debug(f"Appending root dir: '{root_dir}' to sys.path")

from stages.vectorizers.Vectorizer import DATA_DIR_PATH
from stages.models.Model import Model

def main():
    dataset_name: str = sys.argv[1]
    datacleaner_name: str = sys.argv[2]
    vectorizer_A: str = sys.argv[3]
    vectorizer_B: str = sys.argv[4]
    vectorizer_name: str = sys.argv[5]

    XA_train, XA_test, yA_train, yA_test, A_metadata = Model.load_train_test(dataset_name, datacleaner_name, vectorizer_A)
    XB_train, XB_test, yB_train, yB_test, B_metadata = Model.load_train_test(dataset_name, datacleaner_name, vectorizer_B)

    X_train = np.vstack((XA_train, XB_train))
    X_test = np.vstack((XA_test, XB_test))

    path = os.path.join(get_root_dir(), DATA_DIR_PATH, dataset_name, f"{datacleaner_name}_{vectorizer_name}")
    if not os.path.exists(path):
        logger.debug(f"Creating output directory {path}")
        os.makedirs(path)

    path = os.path.join(path, "data")
    np.savez_compressed(path, X_train=X_train, X_test=X_test, y_train=yA_train, y_test=yA_test, metadata = np.array({}))




