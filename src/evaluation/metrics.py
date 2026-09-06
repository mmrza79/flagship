"""Metrics for multiclass gait-phase predictions on held-out subjects."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: ArrayLike | None = None,
) -> dict[str, Any]:
    """Compute aggregate and per-class metrics for multiclass classification.

    Undefined per-class precision/recall values are reported as zero. The result
    is JSON serializable to support reproducible experiment artifacts.
    """

    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)
    if true_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional")
    if true_values.size == 0 or true_values.size != predicted_values.size:
        raise ValueError("y_true and y_pred must have the same non-zero length")

    class_labels = np.asarray(labels) if labels is not None else np.unique(
        np.concatenate([true_values, predicted_values])
    )
    if class_labels.ndim != 1 or class_labels.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence")

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_values,
        predicted_values,
        labels=class_labels,
        average="macro",
        zero_division=0,
    )
    per_class_precision, per_class_recall, per_class_f1, support = (
        precision_recall_fscore_support(
            true_values,
            predicted_values,
            labels=class_labels,
            average=None,
            zero_division=0,
        )
    )

    return {
        "accuracy": float(accuracy_score(true_values, predicted_values)),
        "balanced_accuracy": float(balanced_accuracy_score(true_values, predicted_values)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "labels": [str(label) for label in class_labels],
        "confusion_matrix": confusion_matrix(
            true_values, predicted_values, labels=class_labels
        ).tolist(),
        "per_class": {
            str(label): {
                "precision": float(precision),
                "recall_sensitivity": float(recall),
                "f1": float(f1),
                "support": int(class_support),
            }
            for label, precision, recall, f1, class_support in zip(
                class_labels,
                per_class_precision,
                per_class_recall,
                per_class_f1,
                support,
                strict=True,
            )
        },
    }

