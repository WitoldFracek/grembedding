import os
from abc import ABC, abstractmethod

import pandas as pd
from loguru import logger

from utils.environment import get_root_dir

DATA_DIR = 'data'
ALL_RAW_DATASETS = 'datasets_raw'
RAW_DIR = 'raw'


class DataLoader(ABC):

    @abstractmethod
    def create_dataset(self) -> None:
        """Method called by DVC in order to create the dataset in the dataloader stage"""
        pass

    @property
    def dataset_name(self) -> str:
        """Name of the dataset for file saving"""
        return self.__class__.__name__

    def _save_dataset(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
        """Utility function to save train/test dfs to parquet files. Expected columns: `text` and `label`"""
        path = self._data_output_dir()

        if not os.path.exists(path):
            logger.debug(f"Creating directory {path}")
            os.makedirs(path)

        train_path = os.path.join(path, "train.parquet")
        test_path = os.path.join(path, "test.parquet")

        logger.info(f"Saving created dataset to {path}")
        df_train.to_parquet(train_path, compression='gzip')
        df_test.to_parquet(test_path, compression='gzip')

    @property
    def raw_datasets_dir(self) -> str | os.PathLike:
        return os.path.join(get_root_dir(), ALL_RAW_DATASETS)

    @staticmethod
    def _get_file_path(filename: str) -> str | os.PathLike:
        return os.path.join(get_root_dir(), ALL_RAW_DATASETS, filename)

    def _data_output_dir(self) -> str | os.PathLike:
        return os.path.join(get_root_dir(), DATA_DIR, self.dataset_name, RAW_DIR)
