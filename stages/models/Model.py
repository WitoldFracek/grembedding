import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any

import mlflow
import numpy as np
from loguru import logger

from utils.environment import get_root_dir

DATA_DIR_PATH = 'data'
TEST_FILENAME = 'test.parquet'
TRAIN_FILENAME = 'train.parquet'


class Model(ABC):

    @abstractmethod
    def evaluate(self, dataset: str, datacleaner: str, vectorizer: str, params_name: str,
                 params: Dict[str, int | float | str]) -> None:
        """
        :dataset: name of dataset
        :datacleaner: name of datacleaner that was used to clean the data
        :vectorizer: name of vectorizer that was used to vectorize the data
        """
        pass

    def load_train_test(self, dataset: str, datacleaner: str, vectorizer: str) -> Tuple[
        np.matrix, np.matrix, np.ndarray, np.ndarray, Any]:
        """
        Load train and test data
        :dataset: name of dataset
        :datacleaner: name of datacleaner that was used to clean the data
        :return: tuple of train and test 
        """
        path = self.get_input_dir(dataset, datacleaner, vectorizer)
        path = os.path.join(path, "data.npz")

        data = np.load(path, allow_pickle=True)

        return data["X_train"], data["X_test"], data["y_train"], data["y_test"], data["metadata"]

    @staticmethod
    def save_mlflow_results(params: Dict[str, str | int | float], metrics: Dict[str, float]) -> None:
        """Saves params & metrics to mlflow"""
        logger.info("Saving results to Mlflow...")
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

    def save_json_results(self, dataset: str, datacleaner: str, vectorizer: str, params_name: str,
                          params: Dict[str, str | int | float], metrics: Dict[str, float]) -> None:
        dirs_in_path = ["results", dataset, datacleaner, vectorizer, self.__class__.__name__]
        path = get_root_dir()

        for p in dirs_in_path:
            path = os.path.join(path, p)
            if not os.path.exists(path):
                os.makedirs(path)

        filename = f"{params_name}.json"
        results = {'params': params, 'metrics': metrics, 'dataset': dataset, 'datacleaner': datacleaner,
                   'vectorizer': vectorizer, 'params_name': params_name}
        with open(os.path.join(path, filename), 'w') as file:
            json.dump(results, file)

    @staticmethod
    def get_input_dir(dataset_name, datacleaner_name, vectorizer_name):
        return os.path.join(get_root_dir(), DATA_DIR_PATH, dataset_name, f"{datacleaner_name}_{vectorizer_name}")
