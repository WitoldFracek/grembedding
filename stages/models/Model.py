import os
from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np
import mlflow
import pandas as pd
import json

from utils.environment import get_root_dir

DATA_DIR_PATH = 'data'
TEST_FILENAME = 'test.parquet'
TRAIN_FILENAME = 'train.parquet'
MLFOW_URI = 'http://127.0.0.1:8080'
MLFLOW_ARTIFACT_PATH = "mlflow_artifacts"


class Model(ABC):

    @abstractmethod
    def evaluate(self, dataset: str, datacleaner: str, vectorizer: str, params_name: str, params: Dict[str, int | float | str]) -> None:
        """
        :dataset: name of dataset
        :datacleaner: name of datacleaner that was used to clean the data
        :vectorizer: name of vectorizer that was used to vectorize the data
        """
        pass

    def load_train_test(self, dataset: str, datacleaner: str, vectorizer: str
                        ) -> Tuple[np.matrix, np.matrix, np.array, np.array]:
        """
        Load train and test data
        :dataset: name of dataset
        :datacleaner: name of datacleaner that was used to clean the data
        :return: tuple of train and test 
        """
        path = self.get_input_dir(dataset, datacleaner, vectorizer)

        X_train_path = os.path.join(path, "X_train.npy")
        X_test_path = os.path.join(path, "X_test.npy")
        y_train_path = os.path.join(path, "y_train.npy")
        y_test_path = os.path.join(path, "y_test.npy")

        X_train = np.load(X_train_path)
        X_test = np.load(X_test_path)
        y_train = np.load(y_train_path)
        y_test = np.load(y_test_path)

        return X_train, X_test, y_train, y_test

    def save_results(self, experiment_name: str, run_name: str, params: Dict[str, str | int | float],
                     metrics: Dict[str, float], clf) -> None:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                sk_model=clf, artifact_path=MLFLOW_ARTIFACT_PATH
            )

    def save_json_results(self, dataset: str, datacleaner: str, vectorizer: str, params_name: str, params: Dict[str, str | int | float], metrics: Dict[str, float]) -> None:
        # results/${item.data.dataset}_${item.data.datacleaner}_${item.data.vectorizer}_${item.model.model}_${item.model.params}.json
        filename = f"{dataset}_{datacleaner}_{vectorizer}_{self.__class__.__name__}_{params_name}.json"
        results = {}
        results['params'] = params
        results['metrics'] = metrics
        results['dataset'] = dataset
        results['datacleaner'] = datacleaner
        results['vectorizer'] = vectorizer
        results['params_name'] = params_name
        with open(os.path.join(get_root_dir(), "results", filename), 'w') as file:
            json.dump(results, file)



    @staticmethod
    def get_input_dir(dataset_name, datacleaner_name, vectorizer_name):
        return os.path.join(get_root_dir(), DATA_DIR_PATH, dataset_name, f"{datacleaner_name}_{vectorizer_name}")