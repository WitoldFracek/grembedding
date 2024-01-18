from typing import Dict

import sklearn.svm as svm
from loguru import logger
from rich.pretty import pretty_repr
from sklearn.preprocessing import StandardScaler

from stages.models.Model import Model
from utils.metrics.classification import compute_classification_metrics
from utils.mlflow.experiments import mlflow_context


class SVC(Model):
    def __init__(self) -> None:
        super().__init__()

    @mlflow_context
    def evaluate(self, dataset: str, datacleaner: str, vectorizer: str, params_name: str,
                 params: Dict[str, int | float | str]) -> None:
        X_train, X_test, y_train, y_test = self.load_train_test(dataset, datacleaner, vectorizer)
        sc = StandardScaler()

        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        logger.info(f"Fit/transform with scaler complete")

        clf = svm.SVC(**params)
        logger.info("Fitting SVC sklearn classifier...")
        logger.debug(pretty_repr(params))

        clf.fit(X_train, y_train)

        logger.info("Predicting with SVC classifier...")
        y_proba = clf.predict_proba(X_test)
        y_pred = y_proba.argmax(axis=1)

        metrics = compute_classification_metrics(y_test, y_pred, y_proba)

        self.save_mlflow_results(params, metrics)
        self.save_json_results(dataset, datacleaner, vectorizer, params_name, params, metrics)
