from typing import Dict
from stages.models.Model import Model
from utils.mlflow.experiments import mlflow_context
import sklearn.cluster as cluster
from utils.metrics.clusterization import compute_clustering_metrics, compute_b_cubed_metrics
from loguru import logger


class KMeans(Model):
    
    @mlflow_context
    def evaluate(self, dataset: str, datacleaner: str, vectorizer: str, params_name: str, params: Dict[str, int | float | str]) -> None:
        X_train, X_test, y_train, y_test, metadata = self.load_train_test(dataset, datacleaner, vectorizer)

        logger.info(f'Fitting KMeans...')
        dbscan = cluster.KMeans(**params)
        labels = dbscan.fit_predict(X_train)
        metrics = compute_clustering_metrics(X_train, labels)
        bcubed = compute_b_cubed_metrics(y_train, labels)
        metrics.update(bcubed)
        self.save_mlflow_results(params, metrics)
        self.save_json_results(dataset, datacleaner, vectorizer, params_name, params, metrics)