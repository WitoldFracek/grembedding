from stages.models.Model import Model
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Literal, Optional, Union
from utils.mlflow.experiments import mlflow_context
from loguru import logger
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
)


class RandomForest(Model):
    
    @mlflow_context
    def evaluate(self, dataset: str, datacleaner: str, vectorizer: str,
                 params_name: str, params: Dict[str, int | float | str]) -> None:
        X_train, X_test, y_train, y_test = self.load_train_test(dataset, datacleaner, vectorizer)
        
        rf = RandomForestClassifier(**params)

        logger.info(f'Fitting sklearn RandomForestClassifier...')
        logger.info(f'Using param: {params}')

        rf.fit(X_train, y_train)

        logger.info("Running predict...")

        y_proba: np.ndarray = rf.predict_proba(X_test)[:, 1]
        y_pred: np.ndarray = rf.predict(X_test)

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
        self.save_mlflow_results(params, metrics)
        self.save_json_results(dataset, datacleaner, vectorizer, params_name, params, metrics=metrics)

        # Classification Report
        logger.info(f"Result F1: {f1:.2f}")



