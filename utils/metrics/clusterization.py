from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import mlflow


def compute_clustering_metrics(x: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if len(np.unique(labels)) <= 1:
        silhouette = -1
        davies_bouldin = -1
        calinski_harabasz = -1
    else:
        silhouette = silhouette_score(x, labels)
        davies_bouldin = davies_bouldin_score(x, labels)
        calinski_harabasz = calinski_harabasz_score(x, labels)
    results = {
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski_harabasz
    }

    logger.info('Preparing plots...')
    for method in ['PCA', 'TSNE', 'UMAP']:
        _log_clusters_plot(x, labels, reduction_method=method)
    
    return results


def _log_clusters_plot(x: np.ndarray, labels: np.ndarray, reduction_method='PCA'):    
    logger.info('Dimensionality reduction...')
    reducer = _get_reducer(reduction_method)
    x_reduced = reducer.fit_transform(x)

    logger.info('Creating plot for clustering...')
    fig, ax = plt.subplots(figsize=(8, 6))

    for label in np.unique(labels):
        cluster = x_reduced[labels == label]
        ax.scatter(cluster[:, 0], cluster[:, 1], label=str(label))
    ax.set_title(f'Clustering\n({reduction_method} reduction)')
    ax.legend()
    mlflow.log_figure(fig, f'clustering_wiht_{reduction_method}.png')


def _get_reducer(name: str):
    reducers = {
        'PCA': PCA(n_components=2),
        'TSNE': TSNE(n_components=2),
        'UMAP': UMAP(n_components=2)
    }
    return reducers[name]