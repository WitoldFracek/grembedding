import os
from typing import Dict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from loguru import logger
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
)
from sklearn.neural_network import MLPClassifier

from stages.models.Model import Model
from utils.experiments import mlflow_context


class MLP(Model):

    @mlflow_context
    def evaluate(self, dataset: str, datacleaner: str, vectorizer: str, params_name: str,
                 params: Dict[str, int | float | str]) -> None:
        X_train, X_test, y_train, y_test = self.load_train_test(dataset, datacleaner, vectorizer)

        clf = MLPClassifier(**params)

        logger.info("Fitting MLP Sklearn base classifier classifier...")
        logger.info(f"Using param: {params}")

        clf.fit(X_train, y_train)

        logger.info("Running predict...")
        # y_pred: np.ndarray = clf.predict(X_test)
        y_proba: np.ndarray = clf.predict_proba(X_test)
        y_pred: np.ndarray = y_proba.argmax(axis=1)

        # Evaluation metrics
        logger.info("Calculating metrics...")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")
        roc_auc = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro"
        )

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }

        # Logging metrics
        logger.info("Saving metrics...")
        self.save_mlflow_results(params, metrics, clf)
        self.save_json_results(dataset, datacleaner, vectorizer, params_name, params, metrics=metrics)

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(conf_matrix)

        # Plot training process
        self.plot_training_process(clf)

        # Classification Report
        logger.info(f"Result F1: {f1:.2f}")

    @staticmethod
    def plot_confusion_matrix(conf_matrix):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        mlflow.log_figure(fig, "confusion_matrix.png")

    @staticmethod
    def plot_training_process(clf: MLPClassifier):
        # make seaborn plot with loss curve and val loss
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
    os.environ["DVC_ROOT"] = "../../"
    mlp.evaluate(
        "MsTweetsV2",
        "TweetNormalizationHashtagSkip",
        "CountVectorizer1000",
        "default",
        {"max_iter": 1}
    )
