from typing import Dict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import sklearn.linear_model as lm
from loguru import logger
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from stages.models.Model import Model
from utils.mlflow.experiments import mlflow_context


class LogisticRegression(Model):

    @mlflow_context
    def evaluate(self, dataset: str, datacleaner: str, vectorizer: str, params_name: str,
                 params: Dict[str, int | float | str]) -> None:
        X_train, X_test, y_train, y_test = self.load_train_test(dataset, datacleaner, vectorizer)

        clf = lm.LogisticRegression(n_jobs=-1, **params)

        logger.info("Fitting Logistic Regression classifier...")
        clf.fit(X_train, y_train)

        logger.info("Running predict...")
        y_pred: np.ndarray = clf.predict(X_test)

        # TODO predict proba
        # y_pred: np.ndarray = (y_proba > 0.5).astype("int")

        # Evaluation metrics
        logger.info("Calculating metrics...")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")
        # roc_auc = auc(*roc_curve(y_test, y_proba)[:2])

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            # "roc_auc": roc_auc
        }

        # Logging metrics
        logger.info("Saving metrics...")
        self.save_mlflow_results(params, metrics)
        self.save_json_results(dataset, datacleaner, vectorizer, params_name, params, metrics=metrics)

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(conf_matrix)

        # ROC Curve
        # self.plot_roc_curve(y_test, y_proba)

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

    def plot_roc_curve(self, y_test, y_proba):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
