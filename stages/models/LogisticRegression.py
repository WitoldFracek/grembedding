import itertools
from typing import Dict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import sklearn.linear_model as lm
from loguru import logger
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)

from stages.models.Model import Model


class LogisticRegression(Model):
    def evaluate(self, dataset: str, datacleaner: str, vectorizer: str, params_name: str,
                 params: Dict[str, int | float | str]) -> None:
        X_train, X_test, y_train, y_test = self.load_train_test(dataset, datacleaner, vectorizer)

        experiment_name = dataset
        run_name = f"{datacleaner}-{vectorizer}-{self.__class__.__name__}"

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):
            clf = lm.LogisticRegression(n_jobs=-1, **params)

            logger.info("Fitting Logistic Regression classifier...")
            clf.fit(X_train, y_train)

            logger.info("Running predict...")
            y_pred: np.ndarray = clf.predict(X_test)

            # TOOD predict proba
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
            mlflow.log_param("params", params)
            mlflow.log_metrics(metrics)
            self.save_json_results(dataset, datacleaner, vectorizer, params_name, params, metrics=metrics)

            mlflow.set_tags(
                {
                    "vectorizer": vectorizer,
                }
            )

            # Confusion Matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            # self.plot_confusion_matrix(conf_matrix)

            # ROC Curve
            # self.plot_roc_curve(y_test, y_proba)

            # Classification Report
            logger.info(f"Result F1: {f1:.2f}")

    def plot_confusion_matrix(self, conf_matrix):
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Negative', 'Positive'])
        plt.yticks(tick_marks, ['Negative', 'Positive'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.grid(False)
        for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
            plt.text(j, i, conf_matrix[i, j],
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
        plt.tight_layout()
        plt.show()

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
