"""Replay validation frames through the label-free environment detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.supervisor.environment_change import (
    EnvironmentChangeDetector,
    make_regime_key,
)
from models.supervisor.environment_features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    extract_environment_features,
)
from src.config.loader import load_config


def load_references(archive_path: Path) -> dict[str, dict[str, np.ndarray]]:
    with np.load(archive_path, allow_pickle=False) as archive:
        features = archive["features"].astype(np.float64)
        feature_names = tuple(str(value) for value in archive["feature_names"])
        context_ids = archive["context_ids"].astype(str)
        schema_version = int(archive["feature_schema_version"])
    if feature_names != FEATURE_NAMES or schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError("Reference feature schema does not match this code version.")
    references: dict[str, dict[str, np.ndarray]] = {}
    references["pooled"] = {
        name: features[:, index]
        for index, name in enumerate(FEATURE_NAMES)
    }
    for context_id in sorted(set(context_ids)):
        selected = features[context_ids == context_id]
        references[context_id] = {
            name: selected[:, index]
            for index, name in enumerate(FEATURE_NAMES)
        }
    return references


def read_frame(sample_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(sample_path, allow_pickle=False) as sample:
        return np.asarray(sample["rx_dd"]), np.asarray(sample["h_hat"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--detector-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "environment_detector_v1.json",
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gain-perturbation", type=float, default=1.5)
    args = parser.parse_args()
    config = load_config(args.config)
    with args.detector_config.open(encoding="utf-8") as file:
        detector_config = json.load(file)

    dataset_root = PROJECT_ROOT / config.dataset.root
    raw_dir = dataset_root / config.dataset.raw_dir
    metadata_path = (
        dataset_root / config.dataset.processed_dir / config.dataset.split_metadata_file
    )
    metadata = pd.read_csv(metadata_path)
    validation = metadata.loc[
        metadata["split"].astype(str).str.lower()
        == str(detector_config["replay_split"]).lower()
    ].reset_index(drop=True)
    if validation.empty:
        raise ValueError("No validation rows were found for detector replay.")

    reference_path = args.reference or (
        PROJECT_ROOT
        / "experiments"
        / "environment_detector"
        / "reference_features_v1.npz"
    )
    references = load_references(reference_path)
    alpha = float(detector_config["alpha"])
    results = []
    for context_id, rows in validation.groupby(
        ["snr_db", "velocity_kmh"], sort=True
    ):
        regime_id = make_regime_key(float(context_id[0]), float(context_id[1]))
        rows = rows.reset_index(drop=True)
        if regime_id not in references:
            raise KeyError(f"No fixed reference exists for validation context {regime_id!r}.")
        context_references = {regime_id: references[regime_id]}
        detector = EnvironmentChangeDetector(
            context_references,
            alpha=alpha,
            seed=args.seed,
            familywise_contexts=False,
        )
        nominal_alarms = []
        nominal_log_wealth = {name: [] for name in FEATURE_NAMES}
        shifted_alarms = []
        shifted_log_wealth = {name: [] for name in FEATURE_NAMES}
        shifted_detection_index = None
        for _, row in rows.iterrows():
            rx_dd, h_hat = read_frame(raw_dir / str(row["file"]))
            features = extract_environment_features(rx_dd, h_hat)
            decision = detector.update(features, regime_id=regime_id)
            nominal_alarms.append(decision.status == "SHIFT_DETECTED")
            for name in FEATURE_NAMES:
                nominal_log_wealth[name].append(
                    float(decision.evidence[name]["portfolio_log_wealth"])
                )

        for frame_index, (_, row) in enumerate(rows.iterrows(), start=1):
            rx_dd, h_hat = read_frame(raw_dir / str(row["file"]))
            shifted_features = extract_environment_features(
                rx_dd, h_hat * float(args.gain_perturbation)
            )
            decision = detector.update(shifted_features, regime_id=regime_id)
            shifted_alarms.append(decision.status == "SHIFT_DETECTED")
            for name in FEATURE_NAMES:
                shifted_log_wealth[name].append(
                    float(decision.evidence[name]["portfolio_log_wealth"])
                )
            if shifted_detection_index is None and shifted_alarms[-1]:
                shifted_detection_index = frame_index

        results.append(
            {
                "snr_db": float(context_id[0]),
                "velocity_kmh": float(context_id[1]),
                "validation_frames": len(rows),
                "nominal_alarm_count": int(sum(nominal_alarms)),
                "nominal_alarm_rate": float(np.mean(nominal_alarms)),
                "synthetic_shift": detector_config["controlled_shift"]["name"],
                "synthetic_shift_alarm_count": int(sum(shifted_alarms)),
                "first_synthetic_shift_alarm_frame": shifted_detection_index,
                "max_nominal_log_wealth_by_score": {
                    name: float(np.max(values))
                    for name, values in nominal_log_wealth.items()
                },
                "max_shifted_log_wealth_by_score": {
                    name: float(np.max(values))
                    for name, values in shifted_log_wealth.items()
                },
                "log_alarm_threshold": float(
                    np.log(1.0 / detector.alpha_per_monitor)
                ),
                "synthetic_test_is_physical_evidence": False,
                "alpha_scope": "per known context; no combined guarantee across regime changes",
                "alpha_per_reference_score_monitor": detector.alpha_per_monitor,
                "alpha_threshold_per_monitor": 1.0 / detector.alpha_per_monitor,
            }
        )

    report = {
        "training_started": False,
        "test_split_used": False,
        "reference_split": detector_config["reference_split"],
        "replay_split": detector_config["replay_split"],
        "frames": len(validation),
        "alpha_per_known_context": alpha,
        "context_score_family_size": len(FEATURE_NAMES),
        "alpha_note": "This replay uses one trusted context reference at a time. The runtime default remains family-wise over all configured references unless explicitly constructed in per-context mode.",
        "feature_names": list(FEATURE_NAMES),
        "samples_per_context_are_small_note": (
            "There are only 15 validation frames per condition; this replay is a smoke check, "
            "not an empirical proof of the nominal false-alarm rate."
        ),
        "conditions": results,
    }
    output_directory = PROJECT_ROOT / "experiments" / "environment_detector"
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "validation_replay_v1.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
