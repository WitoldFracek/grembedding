from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import pandas as pd
import os
import pathlib

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
        path = os.path.join(self._get_main_dir_path(), DATA_DIR_PATH, dataset, RAW_DIR)
        train_path = os.path.join(path, TRAIN_FILENAME)
        test_path = os.path.join(path, TEST_FILENAME)
        df_train = pd.read_csv(train_path, sep=';', header=0, names = ["text", "label"])
        df_test = pd.read_csv(test_path, sep=';', header=0, names = ["text", "label"])
        return df_train, df_test
    
    def safe_dataframe_as_parquet(self, dataset: str, df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
        """
        Save dataframes to parquet file
        :dataset: name of dataset
        :df_train: train dataframe
        :df_test: test dataframe
        """
        path = os.path.join(self._get_main_dir_path(), DATA_DIR_PATH, dataset, self.__class__.__name__)
        if not os.path.exists(path):
            os.makedirs(path)
            
        train_path = os.path.join(path, "train.parquet")
        test_path = os.path.join(path, "test.parquet")
        df_train.to_parquet(train_path)
        df_test.to_parquet(test_path)

    def _get_main_dir_path(self) -> str:
        return pathlib.Path(__file__).parent.parent.parent
