"""Reusable offline preprocessing operations for surface EMG signals."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import filtfilt, iirnotch, sosfiltfilt, butter


FloatArray = NDArray[np.float64]


def _as_finite_float_array(signal: ArrayLike, axis: int) -> tuple[FloatArray, int]:
    """Validate signal values and normalize the requested axis."""

    values = np.asarray(signal, dtype=float)
    if values.ndim == 0:
        raise ValueError("signal must contain at least one dimension")
    if not isinstance(axis, int) or not -values.ndim <= axis < values.ndim:
        raise ValueError(f"axis {axis!r} is out of bounds for a {values.ndim}D signal")
    normalized_axis = axis % values.ndim
    if values.shape[normalized_axis] < 2:
        raise ValueError("signal must contain at least two samples along axis")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal must contain only finite values")
    return values, normalized_axis


def _validate_sampling_frequency(sampling_frequency_hz: float) -> float:
    """Validate a sampling frequency supplied from verified metadata/config."""

    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0:
        raise ValueError("sampling_frequency_hz must be a positive finite value")
    return float(sampling_frequency_hz)


def remove_dc(signal: ArrayLike, axis: int = -1) -> FloatArray:
    """Remove each channel's mean (DC offset) along the sample axis."""

    values, normalized_axis = _as_finite_float_array(signal, axis)
    return values - np.mean(values, axis=normalized_axis, keepdims=True)


def bandpass_filter(
    signal: ArrayLike,
    sampling_frequency_hz: float,
    lowcut_hz: float,
    highcut_hz: float,
    order: int = 4,
    axis: int = -1,
) -> FloatArray:
    """Apply an offline zero-phase Butterworth band-pass filter.

    ``sosfiltfilt`` avoids phase displacement and is useful for retrospective
    analysis. It uses future samples, so it is not equivalent to a causal
    real-time implementation.
    """

    values, normalized_axis = _as_finite_float_array(signal, axis)
    fs = _validate_sampling_frequency(sampling_frequency_hz)
    nyquist_hz = fs / 2.0
    if not (0 < lowcut_hz < highcut_hz < nyquist_hz):
        raise ValueError(
            "cutoffs must satisfy 0 < lowcut_hz < highcut_hz < Nyquist "
            f"({nyquist_hz:g} Hz)"
        )
    if not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")

    sos = butter(order, [lowcut_hz, highcut_hz], btype="bandpass", fs=fs, output="sos")
    try:
        return sosfiltfilt(sos, values, axis=normalized_axis)
    except ValueError as exc:
        raise ValueError(
            "signal is too short for zero-phase band-pass filtering; use a longer "
            "record or lower filter order"
        ) from exc


def notch_filter(
    signal: ArrayLike,
    sampling_frequency_hz: float,
    notch_frequency_hz: float,
    quality_factor: float = 30.0,
    axis: int = -1,
) -> FloatArray:
    """Apply an offline zero-phase IIR notch filter at a configured frequency."""

    values, normalized_axis = _as_finite_float_array(signal, axis)
    fs = _validate_sampling_frequency(sampling_frequency_hz)
    nyquist_hz = fs / 2.0
    if not np.isfinite(notch_frequency_hz) or not 0 < notch_frequency_hz < nyquist_hz:
        raise ValueError(f"notch_frequency_hz must lie between 0 and Nyquist ({nyquist_hz:g} Hz)")
    if not np.isfinite(quality_factor) or quality_factor <= 0:
        raise ValueError("quality_factor must be a positive finite value")

    numerator, denominator = iirnotch(notch_frequency_hz, quality_factor, fs=fs)
    try:
        return filtfilt(numerator, denominator, values, axis=normalized_axis)
    except ValueError as exc:
        raise ValueError(
            "signal is too short for zero-phase notch filtering; use a longer record"
        ) from exc


def rectify_signal(signal: ArrayLike) -> FloatArray:
    """Full-wave rectify a finite signal."""

    values = np.asarray(signal, dtype=float)
    if values.ndim == 0 or values.size == 0:
        raise ValueError("signal must be a non-empty array")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal must contain only finite values")
    return np.abs(values)


def normalize_signal(
    signal: ArrayLike,
    method: Literal["none", "zscore", "minmax", "maxabs"] = "none",
    axis: int = -1,
) -> FloatArray:
    """Normalize signal along an axis with explicit constant-signal handling.

    For model evaluation, any learned normalization must be fitted within each
    training fold. This function is intended for explicitly chosen signal-level
    normalization, not global dataset scaling before cross-validation.
    """

    values, normalized_axis = _as_finite_float_array(signal, axis)
    if method == "none":
        return values.copy()
    if method == "zscore":
        center = np.mean(values, axis=normalized_axis, keepdims=True)
        scale = np.std(values, axis=normalized_axis, keepdims=True)
    elif method == "minmax":
        minimum = np.min(values, axis=normalized_axis, keepdims=True)
        center = minimum
        scale = np.max(values, axis=normalized_axis, keepdims=True) - minimum
    elif method == "maxabs":
        center = 0.0
        scale = np.max(np.abs(values), axis=normalized_axis, keepdims=True)
    else:
        raise ValueError("method must be one of: none, zscore, minmax, maxabs")

    safe_scale = np.where(scale == 0, 1.0, scale)
    return (values - center) / safe_scale
