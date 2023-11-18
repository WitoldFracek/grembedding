import os
from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
from loguru import logger

from utils.environment import get_root_dir

DATA_DIR = 'data'
ALL_RAW_DATASETS = 'all_raw_datasets'
RAW_DIR = 'raw'

class DataLoader(ABC):

    @abstractmethod
    def create_dataset(self) -> None:
        pass

    def save_dataset(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
        path = self.data_output_dir()

        if not os.path.exists(path):
            logger.debug(f"Creating directory {path}")
            os.makedirs(path)

        train_path = os.path.join(path, "train.parquet")
        test_path = os.path.join(path, "test.parquet")

        df_train.to_parquet(train_path)
        df_test.to_parquet(test_path)

    @staticmethod
    def get_file_path(filename: str) -> str | os.PathLike:
        return os.path.join(get_root_dir(), DATA_DIR, ALL_RAW_DATASETS, filename)

    def data_output_dir(self) -> str | os.PathLike:
        return os.path.join(get_root_dir(), DATA_DIR, self.__class__.__name__, RAW_DIR)