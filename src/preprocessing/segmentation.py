"""Window segmentation for sampled signals."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def segment_signal(
    signal: ArrayLike,
    window_size: int,
    overlap: int = 0,
    axis: int = 0,
    drop_incomplete: bool = True,
) -> NDArray[np.float64]:
    """Split a signal into windows, returning window index as first dimension.

    ``overlap`` is specified in samples. Overlapping windows from one participant
    must never be split across train and test sets; assign folds by participant
    before interpreting windows as independent examples.
    """

    values = np.asarray(signal, dtype=float)
    if values.ndim == 0 or values.size == 0:
        raise ValueError("signal must be a non-empty array")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal must contain only finite values")
    if not isinstance(axis, int) or not -values.ndim <= axis < values.ndim:
        raise ValueError(f"axis {axis!r} is out of bounds for a {values.ndim}D signal")
    normalized_axis = axis % values.ndim
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if not isinstance(overlap, int) or overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must be an integer in [0, window_size)")

    sample_first = np.moveaxis(values, normalized_axis, 0)
    n_samples = sample_first.shape[0]
    step = window_size - overlap
    starts = list(range(0, max(n_samples - window_size + 1, 0), step))

    if not drop_incomplete and (not starts or starts[-1] + window_size < n_samples):
        final_start = starts[-1] + step if starts else 0
        starts.append(final_start)

    windows: list[NDArray[np.float64]] = []
    for start in starts:
        window = sample_first[start : start + window_size]
        if window.shape[0] < window_size:
            pad_width = [(0, window_size - window.shape[0])] + [(0, 0)] * (window.ndim - 1)
            window = np.pad(window, pad_width, mode="constant", constant_values=np.nan)
        windows.append(window)

    output_shape = (0, window_size, *sample_first.shape[1:])
    return np.stack(windows) if windows else np.empty(output_shape, dtype=float)
