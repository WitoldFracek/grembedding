from typing import Optional

import mlflow
import numpy as np
from loguru import logger
from numpy import interp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    roc_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns


def compute_classification_metrics(y_test: np.ndarray, y_pred: np.ndarray,
                                   y_proba: Optional[np.ndarray] = None) -> dict[str, float]:
    """Computes classification metrics

    Args:
        y_test (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_proba (Optional[np.ndarray], optional): Predicted probabilities. Defaults to None.

    Returns:
        dict[str, float]: Dictionary with metrics. Saves confusion matrix and ROC AUC curve to MLflow.
    """
    logger.info("Calculating metrics...")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    # TODO: fix this - multiclass?
    # if y_proba is not None:
    #     # Only if y_proba is supported
    #     roc_auc = roc_auc_score(
    #         y_test, y_proba, multi_class="ovr", average="macro"
    #     )
    #     results["roc_auc"] = roc_auc
    #     _log_roc_auc_macro_plot(y_test, y_proba)

    conf_matrix = confusion_matrix(y_test, y_pred)
    _log_confusion_matrix_plot(conf_matrix)

    logger.info(f"Result metrics: F1 = {results['f1_score']:.3f}")
    return results


def _log_confusion_matrix_plot(conf_matrix):
    """Plot confusion matrix."""
    logger.info("Creating confusion matrix...")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    mlflow.log_figure(fig, "confusion_matrix.png")


def _log_roc_auc_macro_plot(y_test, y_proba):
    """Plot macro-average ROC AUC curve."""
    logger.info("Creating macro-average ROC AUC curve...")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_proba.shape[1]

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_proba[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    ax.plot(fpr["macro"], tpr["macro"],
            label='Macro-average ROC curve (area = {0:0.2f})'
                  ''.format(roc_auc["macro"]))

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title('Macro-average ROC AUC curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    mlflow.log_figure(fig, "roc_auc_macro.png")
