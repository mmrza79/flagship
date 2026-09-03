"""Conservative dataset discovery and loading.

No file naming convention or schema is assumed. Dataset-specific adaptation
belongs in :func:`load_dataset` after the real dataset and its documentation are
available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import whosmat


SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt", ".npy", ".npz", ".mat"}
SUBJECT_HINTS = ("subject", "participant", "patient", "person", "subj")
TIME_HINTS = ("time", "timestamp", "sample")
LABEL_HINTS = ("label", "phase", "event", "class", "target")
FREQUENCY_HINTS = ("sampling", "frequency", "sample_rate", "fs")
EMG_HINTS = ("emg", "muscle")


@dataclass(frozen=True)
class DatasetInspection:
    """Serializable summary of files and cautiously inferred schema hints."""

    raw_directory: str
    file_count: int
    directory_count: int
    directories: list[str]
    formats: dict[str, int]
    files: list[dict[str, Any]]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return the inspection as a JSON-serializable dictionary."""

        return asdict(self)


def _matching_columns(columns: list[str], hints: tuple[str, ...]) -> list[str]:
    """Return column names containing any case-insensitive schema hint."""

    return [name for name in columns if any(hint in name.lower() for hint in hints)]


def _inspect_tabular(path: Path, separator: str | None = None) -> dict[str, Any]:
    """Inspect a delimited text file without loading its full contents."""

    read_options: dict[str, Any] = {"nrows": 5}
    if separator is not None:
        read_options["sep"] = separator
    elif path.suffix.lower() == ".txt":
        read_options.update({"sep": None, "engine": "python"})

    preview = pd.read_csv(path, **read_options)
    columns = [str(column) for column in preview.columns]
    return {
        "columns": columns,
        "preview_rows": len(preview),
        "subject_candidates": _matching_columns(columns, SUBJECT_HINTS),
        "emg_candidates": _matching_columns(columns, EMG_HINTS),
        "label_candidates": _matching_columns(columns, LABEL_HINTS),
        "time_candidates": _matching_columns(columns, TIME_HINTS),
        "sampling_frequency_candidates": _matching_columns(columns, FREQUENCY_HINTS),
    }


def _inspect_numpy(path: Path) -> dict[str, Any]:
    """Inspect NumPy array keys, shapes, and dtypes without fabricating meaning."""

    loaded = np.load(path, allow_pickle=False, mmap_mode="r" if path.suffix == ".npy" else None)
    try:
        if isinstance(loaded, np.ndarray):
            return {"shape": list(loaded.shape), "dtype": str(loaded.dtype)}
        return {
            "arrays": {
                key: {"shape": list(loaded[key].shape), "dtype": str(loaded[key].dtype)}
                for key in loaded.files
            }
        }
    finally:
        if hasattr(loaded, "close"):
            loaded.close()


def _inspect_mat(path: Path) -> dict[str, Any]:
    """Inspect MATLAB variable names, shapes, and dtypes."""

    return {
        "variables": {
            name: {"shape": list(shape), "matlab_class": matlab_class}
            for name, shape, matlab_class in whosmat(path)
        }
    }


def inspect_raw_dataset(raw_directory: str | Path) -> DatasetInspection:
    """Inspect raw files recursively without changing or interpreting them.

    Unknown or unsupported files are listed but not opened. Errors for individual
    files are captured in the report so one corrupt file does not hide the rest.
    """

    raw_path = Path(raw_directory)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_path}")
    if not raw_path.is_dir():
        raise NotADirectoryError(f"Expected a directory, received: {raw_path}")

    directories = sorted(path for path in raw_path.rglob("*") if path.is_dir())
    paths = sorted(path for path in raw_path.rglob("*") if path.is_file() and path.name != ".gitkeep")
    formats: dict[str, int] = {}
    file_reports: list[dict[str, Any]] = []

    for path in paths:
        suffix = path.suffix.lower() or "<no extension>"
        formats[suffix] = formats.get(suffix, 0) + 1
        report: dict[str, Any] = {
            "path": path.relative_to(raw_path).as_posix(),
            "format": suffix,
            "size_bytes": path.stat().st_size,
        }
        try:
            if suffix == ".csv":
                report.update(_inspect_tabular(path))
            elif suffix == ".tsv":
                report.update(_inspect_tabular(path, separator="\t"))
            elif suffix == ".txt":
                report.update(_inspect_tabular(path))
            elif suffix in {".npy", ".npz"}:
                report.update(_inspect_numpy(path))
            elif suffix == ".mat":
                report.update(_inspect_mat(path))
            else:
                report["status"] = "listed_only_unsupported_format"
        except Exception as exc:  # inspection must report file-level failures
            report["inspection_error"] = f"{type(exc).__name__}: {exc}"
        file_reports.append(report)

    notes: list[str] = []
    if not paths:
        notes.append("No raw dataset files found. Dataset structure and metadata remain unknown.")
    if any(path.suffix.lower() not in SUPPORTED_SUFFIXES for path in paths):
        notes.append("Some formats were listed but not parsed; add a verified dataset-specific reader.")
    notes.append("Candidate fields are name-based hints only and require verification against dataset documentation.")

    return DatasetInspection(
        raw_directory=str(raw_path),
        file_count=len(paths),
        directory_count=len(directories),
        directories=[path.relative_to(raw_path).as_posix() for path in directories],
        formats=formats,
        files=file_reports,
        notes=notes,
    )


def load_dataset(raw_directory: str | Path) -> pd.DataFrame:
    """Load the research dataset after its schema has been verified.

    This intentionally fails until dataset-specific file joining, participant ID,
    channel, timestamp, label, and sampling-frequency rules are documented.
    """

    inspection = inspect_raw_dataset(raw_directory)
    if inspection.file_count == 0:
        raise FileNotFoundError(
            "No dataset files found in data/raw. Place the original dataset there, "
            "then run scripts/inspect_dataset.py before implementing an adapter."
        )
    raise NotImplementedError(
        "Dataset files exist, but no verified schema adapter is configured. TODO: "
        "implement loading only after confirming participant IDs, sEMG channels, "
        "timestamps, sampling frequency, and gait-phase annotations."
    )
