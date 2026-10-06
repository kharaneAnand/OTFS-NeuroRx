"""Offline validation of target-free receiver reliability scores."""

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

from models.supervisor.reliability_detector import (
    RELIABILITY_FEATURE_NAMES,
    ReceiverReliabilityReference,
)


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    return pearson(average_ranks(left), average_ranks(right))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path)
    args = parser.parse_args()
    output_directory = PROJECT_ROOT / "experiments" / "reliability_detector"
    features_path = args.features or (output_directory / "train_validation_features.csv")
    with (output_directory / "references.json").open(encoding="utf-8") as file:
        reference_payload = json.load(file)
    frame = pd.read_csv(features_path)
    validation = frame.loc[
        frame["split"].astype(str).str.lower() == "validation"
    ].copy()
    if validation.empty:
        raise ValueError("Validation features are empty.")

    rows = []
    receiver_summaries = {}
    for receiver, reference_values in reference_payload["receivers"].items():
        receiver_rows = validation.loc[validation["receiver"] == receiver].copy()
        reference = ReceiverReliabilityReference.from_dict(reference_values)
        feature_matrix = receiver_rows.loc[
            :, list(RELIABILITY_FEATURE_NAMES)
        ].to_numpy(dtype=float)
        receiver_rows["reliability_distance"] = [
            reference.score(vector) for vector in feature_matrix
        ]
        confidence = receiver_rows["receiver_qpsk_margin"].to_numpy(dtype=float)
        environment_distance = receiver_rows["environment_distance"].to_numpy(dtype=float)
        distances = receiver_rows["reliability_distance"].to_numpy(dtype=float)
        ber = receiver_rows["ber"].to_numpy(dtype=float)
        ser = receiver_rows["ser"].to_numpy(dtype=float)
        nmse = receiver_rows["nmse"].to_numpy(dtype=float)
        receiver_summaries[receiver] = {
            "validation_samples": len(receiver_rows),
            "confidence_vs_ber_pearson": pearson(confidence, ber),
            "confidence_vs_ber_spearman": spearman(confidence, ber),
            "environment_distance_vs_ber_pearson": pearson(environment_distance, ber),
            "environment_distance_vs_ber_spearman": spearman(environment_distance, ber),
            "environment_distance_vs_ser_spearman": spearman(environment_distance, ser),
            "environment_distance_vs_nmse_spearman": spearman(environment_distance, nmse),
            "distance_vs_ber_pearson": pearson(distances, ber),
            "distance_vs_ber_spearman": spearman(distances, ber),
            "distance_vs_ser_spearman": spearman(distances, ser),
            "distance_vs_nmse_spearman": spearman(distances, nmse),
            "confidence_mean_low_ber_quartile": float(
                confidence[ber <= np.quantile(ber, 0.25)].mean()
            ),
            "confidence_mean_high_ber_quartile": float(
                confidence[ber >= np.quantile(ber, 0.75)].mean()
            ),
            "distance_mean_low_ber_quartile": float(
                distances[ber <= np.quantile(ber, 0.25)].mean()
            ),
            "distance_mean_high_ber_quartile": float(
                distances[ber >= np.quantile(ber, 0.75)].mean()
            ),
        }
        reference_confidence_mean = float(reference.mean[0])
        reference_confidence_std = float(reference.standard_deviation[0])
        receiver_rows["confidence_only_z_score"] = (
            confidence - reference_confidence_mean
        ) / reference_confidence_std
        receiver_summaries[receiver]["confidence_only_z_score_vs_ber_spearman"] = spearman(
            receiver_rows["confidence_only_z_score"].to_numpy(dtype=float), ber
        )
        rows.extend(receiver_rows.to_dict(orient="records"))

    result = {
        "reference_split": reference_payload["reference_split"],
        "evaluation_split": reference_payload["evaluation_split"],
        "receiver_checkpoints_reused": reference_payload["receiver_checkpoints_reused"],
        "retraining_performed": reference_payload["retraining_performed"],
        "live_features": list(RELIABILITY_FEATURE_NAMES),
        "offline_labels_used_only_for_validation": ["ber", "ser", "nmse"],
        "correlation_direction_expected": {
            "confidence_vs_ber": "negative is desirable",
            "environment_distance_vs_ber": "positive would be desirable, but is tested directly",
            "confidence_only_z_score_vs_ber": "negative is desirable because higher confidence is better",
            "distance_vs_ber": "positive is desirable",
        },
        "correlations_by_receiver": receiver_summaries,
        "interpretation": (
            "Weak or near-zero correlations are reported as findings. They mean the "
            "two-feature reliability signal is not yet predictive for that receiver; "
            "they are not treated as implementation failures."
        ),
    }
    with (output_directory / "validation_results.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, indent=2, allow_nan=False)
    pd.DataFrame(rows).to_csv(
        output_directory / "validation_scores.csv", index=False
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
