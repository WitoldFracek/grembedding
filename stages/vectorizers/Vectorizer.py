import os
from abc import ABC, abstractmethod
from typing import Tuple
from loguru import logger
import numpy as np
import pandas as pd

from utils.environment import get_root_dir

DATA_DIR_PATH = 'data'
TEST_FILENAME = 'test.parquet'
TRAIN_FILENAME = 'train.parquet'


class Vectorizer(ABC):

    @abstractmethod
    def vectorize(self, dataset: str, datacleaner: str) -> None:
        """
        :dataset: name of dataset
        :datacleaner: name of datacleaner that was used to clean the data
        """
        pass

    def load_train_test_dataframes(self, dataset: str, datacleaner: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load train and test data
        :dataset: name of dataset
        :datacleaner: name of datacleaner that was used to clean the data
        :return: tuple of train and test 
        """
        path = self.get_input_dir(dataset, datacleaner)
        train_path = os.path.join(path, TRAIN_FILENAME)
        test_path = os.path.join(path, TEST_FILENAME)

        df_train = pd.read_parquet(train_path)
        df_test = pd.read_parquet(test_path)

        return df_train, df_test

    def save_as_npy(self, dataset: str, datacleaner: str, X_train: np.matrix, X_test: np.matrix, y_train: np.array, y_test: np.array) -> None:
        path = self.get_output_dir(dataset, datacleaner)

        if not os.path.exists(path):
            logger.debug(f"Creating output directory {path}")
            os.makedirs(path)
        
        path = os.path.join(path, "data")
        np.savez_compressed(path, X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test)

        # X_train_path = os.path.join(path, "X_train")
        # X_test_path = os.path.join(path, "X_test")
        # y_train_path = os.path.join(path, "y_train")
        # y_test_path = os.path.join(path, "y_test")

        # np.save(X_train_path, X_train)
        # np.save(X_test_path, X_test)
        # np.save(y_train_path, y_train)
        # np.save(y_test_path, y_test)

    @staticmethod
    def get_input_dir(dataset: str, datacleaner: str) -> str | os.PathLike:
        return os.path.join(get_root_dir(), DATA_DIR_PATH, dataset, datacleaner)

    def get_output_dir(self, dataset: str, datacleaner: str) -> str | os.PathLike:
        return os.path.join(
            get_root_dir(), DATA_DIR_PATH, dataset, f"{datacleaner}_{self.__class__.__name__}"
        )
