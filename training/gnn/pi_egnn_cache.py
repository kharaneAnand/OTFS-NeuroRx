"""Persistent analytical-feature cache shared by PI-EGNN dataset splits."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from models.gnn.pi_egnn import oamp_prior_trace
from src.receivers.mmse import build_data_observation, mmse_detect

FEATURE_CACHE_SCHEMA_VERSION = 2
FEATURE_CACHE_KEYS = {
    "cache_schema_version",
    "sample_files",
    "prior_iterations",
    "y_data",
    "mmse_estimate",
    "linear_estimate",
    "prior_estimate",
    "effective_variance",
    "feature_compute_seconds",
    "observation_build_seconds",
    "mmse_seconds",
    "oamp_prior_seconds",
    "numpy_solve_seconds",
    "oamp_operator_setup_seconds",
    "oamp_iteration_seconds",
}


def load_all_split_metadata(metadata_path: str | Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    expected = {"train", "validation", "test"}
    observed = set(metadata["split"].astype(str).str.lower())
    if observed != expected:
        raise ValueError(f"Expected exactly the train/validation/test splits; found {observed}.")
    return metadata.reset_index(drop=True)


def read_locked_prior_iterations(project_root: Path) -> int:
    settings_path = project_root / "experiments" / "pi_egnn" / "prior_config.json"
    with settings_path.open(encoding="utf-8") as file:
        settings = json.load(file)
    iterations = int(settings["oamp_prior_iterations"])
    if iterations <= 0:
        raise ValueError("Locked OAMP prior iteration count must be positive.")
    return iterations


def _cache_index(cache: dict[str, np.ndarray], sample_file: str) -> int:
    indices = np.flatnonzero(cache["sample_files"] == sample_file)
    if len(indices) != 1:
        raise KeyError(f"Expected exactly one cached row for {sample_file}.")
    return int(indices[0])


def _on_the_fly_features(
    row: pd.Series,
    raw_dir: Path,
    config,
    iterations: int,
) -> dict[str, object]:
    with np.load(raw_dir / str(row["file"]), allow_pickle=False) as sample:
        rx_dd = np.asarray(sample["rx_dd"])
        h_hat = np.asarray(sample["h_hat"])
    noise_power = 10.0 ** (-float(row["snr_db"]) / 10.0)
    feature_start = time.perf_counter()
    observation_start = time.perf_counter()
    y_data = build_data_observation(rx_dd, config)
    observation_seconds = time.perf_counter() - observation_start
    mmse_start = time.perf_counter()
    mmse_estimate = mmse_detect(y_data, h_hat, noise_power)
    mmse_seconds = time.perf_counter() - mmse_start
    prior_start = time.perf_counter()
    prior = oamp_prior_trace(y_data, h_hat, noise_power, iterations=iterations)[-1]
    prior_seconds = time.perf_counter() - prior_start
    return {
        "y_data": y_data.astype(np.complex64),
        "mmse_estimate": mmse_estimate.astype(np.complex64),
        "linear_estimate": prior.linear_estimate.astype(np.complex64),
        "prior_estimate": prior.estimate.astype(np.complex64),
        "effective_variance": np.float32(prior.effective_variance),
        "observation_build_seconds": np.float32(observation_seconds),
        "mmse_seconds": np.float32(mmse_seconds),
        "oamp_prior_seconds": np.float32(prior_seconds),
        "feature_compute_seconds": np.float32(time.perf_counter() - feature_start),
        "numpy_solve_seconds": np.float32(prior.linear_solve_seconds),
        "oamp_operator_setup_seconds": np.float32(prior.operator_setup_seconds),
        "oamp_iteration_seconds": np.float32(prior.iterations_seconds),
    }


def verify_feature_cache(
    metadata: pd.DataFrame,
    raw_dir: Path,
    config,
    cache: dict[str, np.ndarray],
    iterations: int,
    sample_count: int = 5,
) -> dict[str, object]:
    checked = min(int(sample_count), len(metadata))
    checked_files = []
    comparisons = (
        "y_data",
        "mmse_estimate",
        "linear_estimate",
        "prior_estimate",
        "effective_variance",
    )
    for _, row in metadata.head(checked).iterrows():
        file_name = str(row["file"])
        index = _cache_index(cache, file_name)
        fresh = _on_the_fly_features(row, raw_dir, config, iterations)
        for key in comparisons:
            if not np.allclose(cache[key][index], fresh[key], rtol=1e-5, atol=1e-6):
                raise AssertionError(f"Cached {key} differs from on-the-fly result for {file_name}.")
        checked_files.append(file_name)
    return {"samples_checked": checked, "files": checked_files, "all_feature_values_match": True}


def load_or_build_feature_cache(
    metadata: pd.DataFrame,
    raw_dir: str | Path,
    config,
    cache_path: str | Path,
    iterations: int,
    verify_samples: int = 0,
) -> tuple[dict[str, np.ndarray], bool]:
    """Load or build the all-split MMSE/OAMP cache; return cache and built flag."""

    raw_dir = Path(raw_dir)
    cache_path = Path(cache_path)
    expected_files = metadata["file"].astype(str).to_numpy(dtype=str)
    cache = None
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as stored:
            if (
                FEATURE_CACHE_KEYS.issubset(stored.files)
                and int(stored["cache_schema_version"]) == FEATURE_CACHE_SCHEMA_VERSION
                and int(stored["prior_iterations"]) == iterations
                and np.array_equal(stored["sample_files"], expected_files)
            ):
                cache = {key: stored[key].copy() for key in stored.files}

    built = cache is None
    if built:
        row_features = [
            _on_the_fly_features(row, raw_dir, config, iterations)
            for _, row in metadata.reset_index(drop=True).iterrows()
        ]
        cache = {
            "sample_files": expected_files,
            "cache_schema_version": np.asarray(
                FEATURE_CACHE_SCHEMA_VERSION, dtype=np.int32
            ),
            "prior_iterations": np.asarray(iterations, dtype=np.int32),
            "y_data": np.stack([item["y_data"] for item in row_features]),
            "mmse_estimate": np.stack([item["mmse_estimate"] for item in row_features]),
            "linear_estimate": np.stack([item["linear_estimate"] for item in row_features]),
            "prior_estimate": np.stack([item["prior_estimate"] for item in row_features]),
            "effective_variance": np.asarray(
                [item["effective_variance"] for item in row_features], dtype=np.float32
            ),
            "feature_compute_seconds": np.asarray(
                [item["feature_compute_seconds"] for item in row_features], dtype=np.float32
            ),
            "observation_build_seconds": np.asarray(
                [item["observation_build_seconds"] for item in row_features], dtype=np.float32
            ),
            "mmse_seconds": np.asarray(
                [item["mmse_seconds"] for item in row_features], dtype=np.float32
            ),
            "oamp_prior_seconds": np.asarray(
                [item["oamp_prior_seconds"] for item in row_features], dtype=np.float32
            ),
            "numpy_solve_seconds": np.asarray(
                [item["numpy_solve_seconds"] for item in row_features], dtype=np.float32
            ),
            "oamp_operator_setup_seconds": np.asarray(
                [item["oamp_operator_setup_seconds"] for item in row_features], dtype=np.float32
            ),
            "oamp_iteration_seconds": np.asarray(
                [item["oamp_iteration_seconds"] for item in row_features], dtype=np.float32
            ),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **cache)

    if verify_samples:
        verification = verify_feature_cache(
            metadata, raw_dir, config, cache, iterations, verify_samples
        )
        print(json.dumps(verification, indent=2))
    return cache, built