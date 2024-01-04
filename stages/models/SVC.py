from typing import Dict

import sklearn.svm as svm
from loguru import logger
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from stages.models.Model import Model
from utils.mlflow.experiments import mlflow_context


class SVC(Model):
    def __init__(self) -> None:
        super().__init__()

    @mlflow_context
    def evaluate(self, dataset: str, datacleaner: str, vectorizer: str, params_name: str,
                 params: Dict[str, int | float | str]) -> None:
        """
        :dataset: name of dataset
        :datacleaner: name of datacleaner that was used to clean the data
        :vectorizer: name of vectorizer that was used to vectorize the data
        :params: params for model
        """
        X_train, X_test, y_train, y_test = self.load_train_test(dataset, datacleaner, vectorizer)
        sc = StandardScaler()

        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        logger.info(f"Fit/transform with scaler complete")

        a = 1 # test only

        clf = svm.SVC(**params)

        logger.info(f"Fitting SVC classifier...")
        clf.fit(X_train[:100], y_train[:100])
        logger.info("Predicting with SVC classifier...")
        y_pred = clf.predict(X_test[:100])

        logger.info(f"Params: {params}, acc: {accuracy_score(y_test[:100], y_pred)}")
        metrics = {"accuracy": accuracy_score(y_test[:100], y_pred)}

        self.save_mlflow_results(
            params=params,
            metrics=metrics,
            clf=clf
        )

        self.save_json_results(dataset, datacleaner, vectorizer, params_name, params, metrics)
