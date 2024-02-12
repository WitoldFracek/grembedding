import os
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
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

    def save_as_npy(self, dataset: str, datacleaner: str, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray,
                    y_test: np.ndarray, metadata: dict[str, Any] = {}) -> None:
        path = self.get_output_dir(dataset, datacleaner)

        if not os.path.exists(path):
            logger.debug(f"Creating output directory {path}")
            os.makedirs(path)

        path = os.path.join(path, "data")
        np.savez_compressed(path, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, metadata = np.array({}))

    @staticmethod
    def get_input_dir(dataset: str, datacleaner: str) -> str | os.PathLike:
        return os.path.join(get_root_dir(), DATA_DIR_PATH, dataset, datacleaner)

    def get_output_dir(self, dataset: str, datacleaner: str) -> str | os.PathLike:
        return os.path.join(
            get_root_dir(), DATA_DIR_PATH, dataset, f"{datacleaner}_{self.__class__.__name__}"
        )
    
    @staticmethod
    def get_vectoriser_data_dir() -> str:
        return os.path.join(
            get_root_dir(), 
            'stages', 
            'vectorizers', 
            'vectorizer_data'
        )
