from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import pandas as pd
import os
import pathlib
import mlflow

DATA_DIR_PATH = 'data'
TEST_FILENAME = 'test.parquet'
TRAIN_FILENAME = 'train.parquet'
MLFOW_URI = 'http://127.0.0.1:8080'
MLFLOW_ARTIFACT_PATH = "mlflow_artifacts"

class Model(ABC):

    @abstractmethod
    def evaluate(self, dataset: str, datacleaner: str, vectorizer: str) -> None:
        """
        :dataset: name of dataset
        :datacleaner: name of datacleaner that was used to clean the data
        :vectorizer: name of vectorizer that was used to vectorize the data
        """
        pass

    def load_train_test(self, dataset: str, datacleaner: str, vectorizer: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load train and test data
        :dataset: name of dataset
        :datacleaner: name of datacleaner that was used to clean the data
        :return: tuple of train and test 
        """
        path = os.path.join(self._get_main_dir_path(), DATA_DIR_PATH, dataset, datacleaner, vectorizer)
        train_path = os.path.join(path, TRAIN_FILENAME)
        test_path = os.path.join(path, TEST_FILENAME)
        df_train = pd.read_parquet(train_path)
        df_test = pd.read_parquet(test_path)
        X_train = df_train['vectorized_text'].values.tolist()
        y_train = df_train['label'].values.tolist()
        X_test = df_test['vectorized_text'].values.tolist()
        y_test = df_test['label'].values.tolist()
        return X_train, X_test, y_train, y_test
    
    def save_results(self, experiment_name: str, run_name: str, params: Dict[str, str | int | float ], metrics: Dict[str, float], clf) -> None:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                sk_model = clf, artifact_path = MLFLOW_ARTIFACT_PATH
            )
        

    def _get_main_dir_path(self) -> str:
        return pathlib.Path(__file__).parent.parent.parent