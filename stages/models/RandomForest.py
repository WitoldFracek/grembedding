from stages.models.Model import Model
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Literal, Optional, Union

from utils.metrics.classification import compute_classification_metrics
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

        # Evaluation metrics
        metrics = compute_classification_metrics(y_test, y_pred, y_proba)

        # Logging metrics
        logger.info("Saving metrics...")
        self.save_mlflow_results(params, metrics)
        self.save_json_results(dataset, datacleaner, vectorizer, params_name, params, metrics=metrics)



