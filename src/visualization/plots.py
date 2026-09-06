"""Small plotting helpers for dataset-backed exploratory analysis."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike


def plot_emg_processing(
    time: ArrayLike,
    signals: Sequence[ArrayLike],
    labels: Sequence[str],
    amplitude_unit: str = "unknown unit",
    title: str = "sEMG processing stages",
) -> tuple[Figure, list[Axes]]:
    """Plot aligned raw/processed traces in separate labelled axes."""

    time_values = np.asarray(time, dtype=float)
    if time_values.ndim != 1 or time_values.size == 0:
        raise ValueError("time must be a non-empty one-dimensional array")
    if len(signals) == 0 or len(signals) != len(labels):
        raise ValueError("signals and labels must have the same non-zero length")

    prepared_signals = [np.asarray(signal, dtype=float) for signal in signals]
    if any(signal.ndim != 1 or signal.size != time_values.size for signal in prepared_signals):
        raise ValueError("each signal must be one-dimensional and aligned with time")
    if not np.all(np.isfinite(time_values)) or any(
        not np.all(np.isfinite(signal)) for signal in prepared_signals
    ):
        raise ValueError("time and signals must contain only finite values")

    figure, axes_array = plt.subplots(
        len(prepared_signals), 1, sharex=True, figsize=(10, 2.5 * len(prepared_signals))
    )
    axes = [axes_array] if isinstance(axes_array, Axes) else list(axes_array)
    for axis, signal, label in zip(axes, prepared_signals, labels, strict=True):
        axis.plot(time_values, signal, linewidth=0.8)
        axis.set_ylabel(f"Amplitude ({amplitude_unit})")
        axis.set_title(str(label))
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(title)
    figure.tight_layout()
    return figure, axes


def plot_class_distribution(
    labels: ArrayLike,
    title: str = "Gait-phase label distribution",
) -> tuple[Figure, Axes]:
    """Plot counts for observed class labels without assuming class order."""

    label_values = np.asarray(labels)
    if label_values.ndim != 1 or label_values.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    classes, counts = np.unique(label_values.astype(str), return_counts=True)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(classes, counts)
    axis.set_title(title)
    axis.set_xlabel("Gait-phase label")
    axis.set_ylabel("Window count")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    return figure, axis

