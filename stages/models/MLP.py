import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.neural_network import MLPClassifier
from rich.pretty import pretty_repr

from config import mlflow
from stages.models.Model import Model
from utils.metrics.classification import compute_classification_metrics
from utils.mlflow.experiments import mlflow_context

import mlflow


class MLP(Model):

    @mlflow_context
    def evaluate(self, dataset: str, datacleaner: str, vectorizer: str, params_name: str,
                 params: Dict[str, int | float | str]) -> None:
        X_train, X_test, y_train, y_test = self.load_train_test(dataset, datacleaner, vectorizer)

        clf = MLPClassifier(**params)

        logger.info("Fitting MLP sklearn classifier...")
        logger.debug(pretty_repr(params))

        clf.fit(X_train, y_train)
        self.plot_training_process(clf)  # Plot training process (learning curve)

        logger.info("Running predict...")
        y_proba: np.ndarray = clf.predict_proba(X_test)
        y_pred: np.ndarray = y_proba.argmax(axis=1)

        # Calculate and save metrics
        metrics = compute_classification_metrics(y_test, y_pred, y_proba)

        self.save_mlflow_results(params, metrics)
        self.save_json_results(dataset, datacleaner, vectorizer, params_name, params, metrics=metrics)

    @staticmethod
    def plot_training_process(clf: MLPClassifier):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(clf.loss_curve_, label="loss")

        if clf.validation_scores_:
            ax.plot(clf.validation_scores_, label="validation_score")

        ax.set_title('Training process')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        mlflow.log_figure(fig, "training_process.png")


if __name__ == "__main__":
    mlp = MLP()
    os.chdir("../../")
    os.environ["DVC_ROOT"] = "."
    mlp.evaluate(
        "RpTweetsXS",
        "LemmatizerSM",
        "CountVectorizer1000",
        "default",
        {"max_iter": 1}
    )
