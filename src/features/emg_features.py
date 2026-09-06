"""Interpretable time- and frequency-domain features for one sEMG window."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import periodogram


TIME_FEATURES = (
    "mav",
    "rms",
    "waveform_length",
    "variance",
    "zero_crossings",
    "slope_sign_changes",
)
FREQUENCY_FEATURES = ("mean_frequency", "median_frequency")
ALL_FEATURES = TIME_FEATURES + FREQUENCY_FEATURES


def _validate_window(window: ArrayLike, minimum_samples: int = 1) -> NDArray[np.float64]:
    """Return a finite one-dimensional floating-point signal window."""

    values = np.asarray(window, dtype=float)
    if values.ndim != 1:
        raise ValueError("window must be one-dimensional; extract each channel separately")
    if values.size < minimum_samples:
        raise ValueError(f"window must contain at least {minimum_samples} samples")
    if not np.all(np.isfinite(values)):
        raise ValueError("window must contain only finite values")
    return values


def mean_absolute_value(window: ArrayLike) -> float:
    """Calculate mean absolute value (MAV)."""

    values = _validate_window(window)
    return float(np.mean(np.abs(values)))


def root_mean_square(window: ArrayLike) -> float:
    """Calculate root mean square (RMS)."""

    values = _validate_window(window)
    return float(np.sqrt(np.mean(np.square(values))))


def waveform_length(window: ArrayLike) -> float:
    """Calculate cumulative absolute difference between consecutive samples."""

    values = _validate_window(window, minimum_samples=2)
    return float(np.sum(np.abs(np.diff(values))))


def signal_variance(window: ArrayLike) -> float:
    """Calculate population variance (``ddof=0``) of a window."""

    values = _validate_window(window)
    return float(np.var(values, ddof=0))


def zero_crossings(window: ArrayLike, threshold: float = 0.0) -> int:
    """Count sign changes whose sample-to-sample amplitude exceeds threshold."""

    values = _validate_window(window, minimum_samples=2)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a non-negative finite value")
    opposite_sign = values[:-1] * values[1:] < 0
    sufficient_change = np.abs(np.diff(values)) >= threshold
    return int(np.count_nonzero(opposite_sign & sufficient_change))


def slope_sign_changes(window: ArrayLike, threshold: float = 0.0) -> int:
    """Count local slope reversals exceeding a noise-suppression threshold."""

    values = _validate_window(window, minimum_samples=3)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a non-negative finite value")
    left_difference = values[1:-1] - values[:-2]
    right_difference = values[1:-1] - values[2:]
    reversals = left_difference * right_difference > 0
    sufficient_change = (np.abs(left_difference) >= threshold) | (
        np.abs(right_difference) >= threshold
    )
    return int(np.count_nonzero(reversals & sufficient_change))


def _power_spectrum(
    window: ArrayLike, sampling_frequency_hz: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate one-sided power spectral density using a periodogram."""

    values = _validate_window(window, minimum_samples=2)
    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0:
        raise ValueError("sampling_frequency_hz must be a positive finite value")
    frequencies, power = periodogram(values, fs=float(sampling_frequency_hz), detrend="constant")
    total_power = float(np.sum(power))
    if total_power <= np.finfo(float).eps:
        raise ValueError("frequency features are undefined for a signal with negligible spectral power")
    return frequencies, power


def mean_frequency(window: ArrayLike, sampling_frequency_hz: float) -> float:
    """Calculate power-weighted mean frequency (MNF)."""

    frequencies, power = _power_spectrum(window, sampling_frequency_hz)
    return float(np.sum(frequencies * power) / np.sum(power))


def median_frequency(window: ArrayLike, sampling_frequency_hz: float) -> float:
    """Calculate frequency dividing periodogram power into equal halves (MDF)."""

    frequencies, power = _power_spectrum(window, sampling_frequency_hz)
    cumulative_power = np.cumsum(power)
    median_index = int(np.searchsorted(cumulative_power, cumulative_power[-1] / 2.0))
    return float(frequencies[min(median_index, frequencies.size - 1)])


def extract_feature_vector(
    window: ArrayLike,
    selected_features: Iterable[str] = ALL_FEATURES,
    sampling_frequency_hz: float | None = None,
    zero_crossing_threshold: float = 0.0,
    slope_sign_threshold: float = 0.0,
) -> dict[str, float]:
    """Extract selected features from one channel window.

    A verified sampling frequency is mandatory when MNF or MDF is selected.
    """

    values = _validate_window(window, minimum_samples=3)
    selected = tuple(selected_features)
    unknown = sorted(set(selected) - set(ALL_FEATURES))
    if unknown:
        raise ValueError(f"unknown features: {', '.join(unknown)}")
    if len(set(selected)) != len(selected):
        raise ValueError("selected_features must not contain duplicates")
    if any(name in FREQUENCY_FEATURES for name in selected) and sampling_frequency_hz is None:
        raise ValueError("sampling_frequency_hz is required for frequency-domain features")

    functions = {
        "mav": lambda: mean_absolute_value(values),
        "rms": lambda: root_mean_square(values),
        "waveform_length": lambda: waveform_length(values),
        "variance": lambda: signal_variance(values),
        "zero_crossings": lambda: float(zero_crossings(values, zero_crossing_threshold)),
        "slope_sign_changes": lambda: float(
            slope_sign_changes(values, slope_sign_threshold)
        ),
        "mean_frequency": lambda: mean_frequency(values, float(sampling_frequency_hz)),
        "median_frequency": lambda: median_frequency(values, float(sampling_frequency_hz)),
    }
    return {name: float(functions[name]()) for name in selected}

