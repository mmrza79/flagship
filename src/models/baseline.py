"""Leakage-aware classical baselines for subject-independent evaluation."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ModelName = Literal["logistic_regression", "svm", "random_forest"]
CVStrategy = Literal["loso", "group_k_fold"]


def build_classifier_pipeline(
    model_name: ModelName,
    random_seed: int = 42,
    model_options: dict[str, Any] | None = None,
) -> Pipeline:
    """Build a classifier with fold-local feature scaling where appropriate."""

    options = dict(model_options or {})
    if model_name == "logistic_regression":
        options.setdefault("max_iter", 1000)
        options.setdefault("random_state", random_seed)
        estimator = LogisticRegression(**options)
        scaler: StandardScaler | str = StandardScaler()
    elif model_name == "svm":
        options.setdefault("kernel", "rbf")
        estimator = SVC(**options)
        scaler = StandardScaler()
    elif model_name == "random_forest":
        options.setdefault("n_estimators", 200)
        options.setdefault("random_state", random_seed)
        options.setdefault("n_jobs", -1)
        estimator = RandomForestClassifier(**options)
        scaler = "passthrough"
    else:
        raise ValueError(
            "model_name must be logistic_regression, svm, or random_forest"
        )
    return Pipeline([("scaler", scaler), ("classifier", estimator)])


def _validated_arrays(
    features: ArrayLike, targets: ArrayLike, groups: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[Any], NDArray[Any]]:
    """Validate aligned feature, target, and participant arrays."""

    feature_values = np.asarray(features, dtype=float)
    target_values = np.asarray(targets)
    group_values = np.asarray(groups)
    if feature_values.ndim != 2 or feature_values.shape[0] == 0:
        raise ValueError("features must be a non-empty 2D matrix")
    if not np.all(np.isfinite(feature_values)):
        raise ValueError("features must contain only finite values")
    if target_values.ndim != 1 or group_values.ndim != 1:
        raise ValueError("targets and groups must be one-dimensional")
    if not (feature_values.shape[0] == target_values.size == group_values.size):
        raise ValueError("features, targets, and groups must contain equal rows")
    if np.unique(target_values).size < 2:
        raise ValueError("targets must contain at least two classes")
    if np.unique(group_values).size < 2:
        raise ValueError("subject-independent evaluation requires at least two groups")
    return feature_values, target_values, group_values


def evaluate_grouped_model(
    pipeline: Pipeline,
    features: ArrayLike,
    targets: ArrayLike,
    groups: ArrayLike,
    strategy: CVStrategy = "loso",
    n_splits: int = 5,
) -> tuple[NDArray[Any], list[dict[str, Any]]]:
    """Generate out-of-fold predictions with participant-disjoint folds.

    Scaling is fitted inside each fold through the supplied scikit-learn
    ``Pipeline``. Callers must ensure all overlapping windows from a recording
    retain the same participant group.
    """

    feature_values, target_values, group_values = _validated_arrays(
        features, targets, groups
    )
    unique_groups = np.unique(group_values)
    if strategy == "loso":
        splitter = LeaveOneGroupOut()
    elif strategy == "group_k_fold":
        if not isinstance(n_splits, int) or not 2 <= n_splits <= unique_groups.size:
            raise ValueError("n_splits must be between 2 and the number of groups")
        splitter = GroupKFold(n_splits=n_splits)
    else:
        raise ValueError("strategy must be loso or group_k_fold")

    predictions = np.empty(target_values.shape, dtype=target_values.dtype)
    fold_records: list[dict[str, Any]] = []
    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(feature_values, target_values, group_values), start=1
    ):
        train_groups = np.unique(group_values[train_indices])
        test_groups = np.unique(group_values[test_indices])
        if np.intersect1d(train_groups, test_groups).size:
            raise RuntimeError("group leakage detected between train and test fold")
        if np.unique(target_values[train_indices]).size < 2:
            raise ValueError(
                f"fold {fold_index} training data contain fewer than two classes"
            )

        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(feature_values[train_indices], target_values[train_indices])
        predictions[test_indices] = fold_pipeline.predict(feature_values[test_indices])
        fold_records.append(
            {
                "fold": fold_index,
                "train_size": int(train_indices.size),
                "test_size": int(test_indices.size),
                "train_groups": [str(group) for group in train_groups],
                "test_groups": [str(group) for group in test_groups],
            }
        )

    return predictions, fold_records

