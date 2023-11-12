from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import pandas as pd
import os
import pathlib

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
        path = os.path.join(self._get_root_dir_path(), DATA_DIR_PATH, dataset, datacleaner)
        train_path = os.path.join(path, TRAIN_FILENAME)
        test_path = os.path.join(path, TEST_FILENAME)
        df_train = pd.read_parquet(train_path)
        df_test = pd.read_parquet(test_path)
        return df_train, df_test
    
    def safe_dataframe_as_parquet(self, dataset: str, datacleaner: str, df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
        """
        Save dataframes to parquet file
        :dataset: name of dataset
        :datacleaner: name of datacleaner that was used to clean the data
        :df_train: train dataframe
        :df_test: test dataframe
        """
        path = os.path.join(self._get_root_dir_path(), DATA_DIR_PATH, dataset, f"{datacleaner}_{self.__class__.__name__}")
        if not os.path.exists(path):
            os.makedirs(path)
            
        train_path = os.path.join(path, TRAIN_FILENAME)
        test_path = os.path.join(path, TEST_FILENAME)
        df_train.to_parquet(train_path)
        df_test.to_parquet(test_path)

    def _get_root_dir_path(self) -> str:
        # TODO mock this env var in non-dvc runs / tests
        return os.environ["DVC_ROOT"]