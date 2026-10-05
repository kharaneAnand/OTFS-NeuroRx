"""Evaluate the stateless simple environment distance on validation data."""

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

from models.supervisor.simple_environment_detector import (
    SIMPLE_FEATURE_NAMES,
    SimpleEnvironmentReference,
    extract_simple_environment_features,
)
from src.config.loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--channel-noise-relative",
        type=float,
        default=0.5,
        help="Complex H_hat corruption RMS relative to H_hat RMS.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.channel_noise_relative < 0.0:
        raise ValueError("--channel-noise-relative must be non-negative.")
    config = load_config(args.config)
    reference_path = PROJECT_ROOT / "experiments" / "environment_detector_simple" / "reference.json"
    with reference_path.open(encoding="utf-8") as file:
        reference = SimpleEnvironmentReference.from_dict(json.load(file))

    dataset_root = PROJECT_ROOT / config.dataset.root
    raw_dir = dataset_root / config.dataset.raw_dir
    metadata_path = dataset_root / config.dataset.processed_dir / config.dataset.split_metadata_file
    metadata = pd.read_csv(metadata_path)
    validation = metadata.loc[
        metadata["split"].astype(str).str.lower() == "validation"
    ].reset_index(drop=True)
    if validation.empty:
        raise ValueError("Validation split is empty.")

    nominal_scores = []
    perturbed_scores = []
    rows = []
    condition_scores: dict[tuple[int, int], list[float]] = {}
    rng = np.random.default_rng(args.seed)
    for _, row in validation.iterrows():
        with np.load(raw_dir / str(row["file"]), allow_pickle=False) as sample:
            rx_dd = np.asarray(sample["rx_dd"])
            h_hat = np.asarray(sample["h_hat"])
        noise_power = 10.0 ** (-float(row["snr_db"]) / 10.0)
        nominal_features = extract_simple_environment_features(
            rx_dd, h_hat, noise_power, config
        )
        h_hat_rms = float(np.sqrt(np.mean(np.abs(h_hat) ** 2)))
        complex_noise = (
            rng.normal(size=h_hat.shape) + 1j * rng.normal(size=h_hat.shape)
        ) * (args.channel_noise_relative * h_hat_rms / np.sqrt(2.0))
        perturbed_features = extract_simple_environment_features(
            rx_dd, h_hat + complex_noise, noise_power, config
        )
        nominal_score = reference.score(nominal_features)
        perturbed_score = reference.score(perturbed_features)
        nominal_scores.append(nominal_score)
        perturbed_scores.append(perturbed_score)
        condition = (int(row["snr_db"]), int(row["velocity_kmh"]))
        condition_scores.setdefault(condition, []).append(nominal_score)
        rows.append(
            {
                "file": str(row["file"]),
                "snr_db": int(row["snr_db"]),
                "velocity_kmh": int(row["velocity_kmh"]),
                "nominal_score": nominal_score,
                "perturbed_score": perturbed_score,
                "score_increase": perturbed_score - nominal_score,
            }
        )

    condition_rows = []
    for (snr_db, velocity_kmh), scores in sorted(condition_scores.items()):
        values = np.asarray(scores, dtype=np.float64)
        condition_rows.append(
            {
                "snr_db": snr_db,
                "velocity_kmh": velocity_kmh,
                "samples": len(values),
                "score_mean": float(np.mean(values)),
                "score_median": float(np.median(values)),
                "score_std": float(np.std(values, ddof=1)),
                "score_min": float(np.min(values)),
                "score_max": float(np.max(values)),
            }
        )

    report = {
        "training_started": False,
        "test_split_used": False,
        "reference_split": "train",
        "evaluation_split": "validation",
        "feature_names": list(SIMPLE_FEATURE_NAMES),
        "distance": "sqrt(mean(((feature - training_mean) / training_std)^2))",
        "channel_noise_relative": args.channel_noise_relative,
        "nominal_score_mean": float(np.mean(nominal_scores)),
        "nominal_score_median": float(np.median(nominal_scores)),
        "perturbed_score_mean": float(np.mean(perturbed_scores)),
        "perturbed_score_median": float(np.median(perturbed_scores)),
        "perturbed_mean_is_higher": bool(
            np.mean(perturbed_scores) > np.mean(nominal_scores)
        ),
        "validation_frames": len(validation),
        "condition_score_rows": condition_rows,
        "feature_dependence_note": (
            "Both features use the same reconstruction residual; estimated_snr_db "
            "also divides by received signal power, while the residual proxy does not. "
            "The real-condition score table is therefore informative but not an independent "
            "validation of two unrelated signals."
        ),
        "interpretation": (
            "This is an independent validation replay because the reference uses training frames; "
            "the perturbation is synthetic and target-free, not a physical-channel result."
        ),
    }
    output_directory = PROJECT_ROOT / "experiments" / "environment_detector_simple"
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "validation_results.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)
    pd.DataFrame(condition_rows).to_csv(
        output_directory / "validation_condition_scores.csv", index=False
    )
    pd.DataFrame(rows).to_csv(
        output_directory / "validation_scores.csv", index=False
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
