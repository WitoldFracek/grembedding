import os
from abc import ABC, abstractmethod
from typing import Tuple
from loguru import logger

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

    def save_dataframe_as_parquet(self, dataset: str, datacleaner: str, df_train: pd.DataFrame,
                                  df_test: pd.DataFrame) -> None:
        """
        Save dataframes to parquet file
        :dataset: name of dataset
        :datacleaner: name of datacleaner that was used to clean the data
        :df_train: train dataframe
        :df_test: test dataframe
        """
        path = self.get_output_dir(dataset, datacleaner)

        if not os.path.exists(path):
            logger.debug(f"Creating output directory {path}")
            os.makedirs(path)

        train_path = os.path.join(path, TRAIN_FILENAME)
        test_path = os.path.join(path, TEST_FILENAME)

        df_train.to_parquet(train_path)
        df_test.to_parquet(test_path)

    @staticmethod
    def get_input_dir(dataset: str, datacleaner: str) -> str | os.PathLike:
        return os.path.join(get_root_dir(), DATA_DIR_PATH, dataset, datacleaner)

    def get_output_dir(self, dataset: str, datacleaner: str) -> str | os.PathLike:
        return os.path.join(
            get_root_dir(), DATA_DIR_PATH, dataset, f"{datacleaner}_{self.__class__.__name__}"
        )
