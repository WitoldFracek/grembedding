from typing import Dict

import numpy as np
import sklearn.linear_model as lm
from loguru import logger
from rich.pretty import pretty_repr

from stages.models.Model import Model
from utils.metrics.classification import compute_classification_metrics
from utils.mlflow.experiments import mlflow_context


class LogisticRegression(Model):

    @mlflow_context
    def evaluate(self, dataset: str, datacleaner: str, vectorizer: str, params_name: str,
                 params: Dict[str, int | float | str]) -> None:
        X_train, X_test, y_train, y_test, metadata = self.load_train_test(dataset, datacleaner, vectorizer)

        clf = lm.LogisticRegression(n_jobs=-1, **params)
        logger.info("Fitting Logistic Regression classifier...")
        logger.debug(pretty_repr(params))

        clf.fit(X_train, y_train)

        logger.info("Running predict...")
        y_proba: np.ndarray = clf.predict_proba(X_test)
        y_pred = y_proba.argmax(axis=1)

        metrics = compute_classification_metrics(y_test, y_pred, y_proba)

        self.save_mlflow_results(params, metrics)
        self.save_json_results(dataset, datacleaner, vectorizer, params_name, params, metrics=metrics)
