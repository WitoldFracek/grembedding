import os
from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
from loguru import logger

from utils.environment import get_root_dir

DATA_DIR_PATH = 'data'
RAW_DIR = 'raw'
TEST_FILENAME = 'test.csv'
TRAIN_FILENAME = 'train.csv'


class DataCleaner(ABC):

    @abstractmethod
    def clean_data(self, dataset: str) -> None:
        """
        :dataset: name of dataset
        :return: dict of train and test data
        """
        pass

    def load_dataset(self, dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load train and test data
        :dataset: name of dataset
        :return: tuple of train and test dataframe
        """
        path = self.data_input_dir(dataset)
        train_path = os.path.join(path, TRAIN_FILENAME)
        test_path = os.path.join(path, TEST_FILENAME)

        df_train = pd.read_csv(train_path, sep=';', header=0, names=["text", "label"])
        df_test = pd.read_csv(test_path, sep=';', header=0, names=["text", "label"])

        return df_train, df_test

    def save_dataframe_as_parquet(self, dataset: str, df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
        """
        Save dataframes to parquet file
        :dataset: name of dataset
        :df_train: train dataframe
        :df_test: test dataframe
        """
        path = self.data_output_dir(dataset)
        if not os.path.exists(path):
            logger.debug(f"Creating directory {path}")
            os.makedirs(path)

        train_path = os.path.join(path, "train.parquet")
        test_path = os.path.join(path, "test.parquet")

        df_train.to_parquet(train_path)
        df_test.to_parquet(test_path)

    @staticmethod
    def data_input_dir(dataset_name: str) -> str | os.PathLike:
        return os.path.join(get_root_dir(), DATA_DIR_PATH, dataset_name, RAW_DIR)

    def data_output_dir(self, dataset_name: str) -> str | os.PathLike:
        return os.path.join(get_root_dir(), DATA_DIR_PATH, dataset_name, self.__class__.__name__)
