"""Evaluate PI-EGNN against the fixed original-GNN and OAMP-DL artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.gnn.pi_egnn import PIEGNN
from src.config.loader import load_config
from training.gnn.pi_egnn_cache import (
    load_all_split_metadata,
    load_or_build_feature_cache,
    read_locked_prior_iterations,
)
from training.gnn.pi_egnn_dataset import PIEGNNGraphDataset, load_split_metadata

METRICS = ("ber", "ser", "nmse")
RECEIVERS = ("pi_egnn", "original_gnn", "oamp_dl")


def metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    predicted_bits = np.stack((prediction.real >= 0, prediction.imag >= 0), axis=-1)
    target_bits = np.stack((target.real >= 0, target.imag >= 0), axis=-1)
    bit_errors = predicted_bits != target_bits
    symbol_errors = np.any(bit_errors, axis=-1)
    return (
        float(np.count_nonzero(bit_errors) / bit_errors.size),
        float(np.count_nonzero(symbol_errors) / target.size),
        float(np.sum(np.abs(prediction - target) ** 2) / np.sum(np.abs(target) ** 2)),
    )


def summarize(values: list[tuple[float, float, float]]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    mean = array.mean(axis=0)
    standard_deviation = array.std(axis=0, ddof=1)
    half_width = 1.96 * standard_deviation / np.sqrt(len(values))
    result = {}
    for index, name in enumerate(METRICS):
        result[name] = float(mean[index])
        result[f"{name}_ci95"] = float(half_width[index])
        result[f"{name}_ci95_low"] = float(mean[index] - half_width[index])
        result[f"{name}_ci95_high"] = float(mean[index] + half_width[index])
    return result


def read_reference(path: Path, receiver: str) -> dict[int, dict[str, object]]:
    frame = pd.read_csv(path)
    if receiver == "original_gnn":
        frame = frame.loc[frame["initialization"] == "mmse"]
    result = {}
    for _, row in frame.iterrows():
        index = int(row["sample_index"])
        result[index] = {
            "snr_db": int(row["snr_db"]),
            "velocity_kmh": int(row["velocity_kmh"]),
            "metrics": tuple(float(row[name]) for name in METRICS)
            if receiver == "original_gnn"
            else tuple(float(row[f"oamp_dl_{name}"]) for name in METRICS),
        }
    return result


def paired_difference(
    candidate: list[tuple[float, float, float]],
    reference: list[tuple[float, float, float]],
) -> dict[str, float]:
    return summarize(
        [tuple(left - right for left, right in zip(a, b)) for a, b in zip(candidate, reference)]
    )


def condition_result(
    condition: tuple[int, int],
    values: dict[str, list[tuple[float, float, float]]],
) -> dict[str, object]:
    result: dict[str, object] = {
        "snr_db": condition[0],
        "velocity_kmh": condition[1],
        "samples": len(values["pi_egnn"]),
    }
    summaries = {receiver: summarize(values[receiver]) for receiver in RECEIVERS}
    for receiver, summary in summaries.items():
        for metric in METRICS:
            result[f"{receiver}_{metric}"] = summary[metric]
            result[f"{receiver}_{metric}_ci95"] = summary[f"{metric}_ci95"]
            result[f"{receiver}_{metric}_ci95_low"] = summary[f"{metric}_ci95_low"]
            result[f"{receiver}_{metric}_ci95_high"] = summary[f"{metric}_ci95_high"]
    for reference in ("original_gnn", "oamp_dl"):
        delta = paired_difference(values["pi_egnn"], values[reference])
        for metric in METRICS:
            result[f"pi_egnn_minus_{reference}_{metric}"] = delta[metric]
            result[f"pi_egnn_minus_{reference}_{metric}_ci95"] = delta[f"{metric}_ci95"]
            result[f"pi_egnn_minus_{reference}_{metric}_ci95_low"] = delta[f"{metric}_ci95_low"]
            result[f"pi_egnn_minus_{reference}_{metric}_ci95_high"] = delta[f"{metric}_ci95_high"]
        result[f"beats_{reference}_all_metrics"] = all(
            summaries["pi_egnn"][metric] < summaries[reference][metric]
            for metric in METRICS
        )
        result[f"paired_ci_supports_{reference}_all_metrics"] = all(
            delta[f"{metric}_ci95_high"] < 0 for metric in METRICS
        )
    return result


def regime_result(
    name: str,
    conditions: list[dict[str, object]],
    reference: str,
) -> dict[str, object]:
    return {
        "conditions": [
            {"snr_db": row["snr_db"], "velocity_kmh": row["velocity_kmh"], "samples": row["samples"]}
            for row in conditions
        ],
        "beats_reference_all_metrics_all_conditions": bool(conditions) and all(
            bool(row[f"beats_{reference}_all_metrics"]) for row in conditions
        ),
        "paired_ci_supports_reference_all_metrics_all_conditions": bool(conditions) and all(
            bool(row[f"paired_ci_supports_{reference}_all_metrics"]) for row in conditions
        ),
        "confidence_intervals": "Per-condition receiver mean CI95 and paired PI-EGNN-minus-reference CI95 are in per_condition_results.csv.",
        "hypothesis": name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    dataset_root = PROJECT_ROOT / config.dataset.root
    raw_dir = dataset_root / config.dataset.raw_dir
    metadata_path = dataset_root / config.dataset.processed_dir / config.dataset.split_metadata_file
    metadata = load_split_metadata(metadata_path, "test")
    prior_iterations = read_locked_prior_iterations(PROJECT_ROOT)
    cache_path = PROJECT_ROOT / "experiments" / "pi_egnn" / "analytical_feature_cache.npz"
    feature_cache, cache_built = load_or_build_feature_cache(
        load_all_split_metadata(metadata_path),
        raw_dir,
        config,
        cache_path,
        prior_iterations,
    )
    print(f"Analytical feature cache {'built' if cache_built else 'reused'}: {cache_path}")

    original_path = PROJECT_ROOT / config.gnn.output.mmse_directory / config.gnn.output.per_sample_file
    oamp_path = PROJECT_ROOT / config.oamp_dl.output.directory / config.oamp_dl.output.per_sample_file
    references = {
        "original_gnn": read_reference(original_path, "original_gnn"),
        "oamp_dl": read_reference(oamp_path, "oamp_dl"),
    }
    expected_indices = set(range(len(metadata)))
    for receiver, rows in references.items():
        if set(rows) != expected_indices:
            raise ValueError(f"{receiver} per-sample artifact does not match the fixed test split.")

    dataset = PIEGNNGraphDataset(metadata, raw_dir, config, feature_cache)
    loader = DataLoader(
        dataset, batch_size=int(config.gnn.training.batch_size), shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    model = PIEGNN(
        observation_count=int(config.representation.expected_channel_shape.rows),
        symbol_count=int(config.representation.expected_channel_shape.cols),
        hidden_features=int(config.gnn.hidden_features),
        layers=int(config.gnn.message_passing_layers),
    ).to(device)
    output_directory = PROJECT_ROOT / "experiments" / "pi_egnn"
    checkpoint = torch.load(output_directory / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Locked OAMP prior iterations: {prior_iterations}")

    grouped: dict[tuple[int, int], dict[str, list[tuple[float, float, float]]]] = {}
    sample_rows = []
    sample_index = 0
    inference_seconds = 0.0
    dataset_and_forward_seconds = 0.0
    with torch.no_grad():
        loader_iterator = iter(loader)
        while True:
            if device.type == "cuda":
                torch.cuda.synchronize()
            sample_start = time.perf_counter()
            try:
                packed, targets = next(loader_iterator)
            except StopIteration:
                break
            if device.type == "cuda":
                torch.cuda.synchronize()
            model_start = time.perf_counter()
            packed_device = packed.to(device)
            targets_device = targets.to(device)
            predictions = model(packed_device).cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize()
            model_elapsed = time.perf_counter() - model_start
            inference_seconds += model_elapsed
            dataset_and_forward_seconds += time.perf_counter() - sample_start
            for position, prediction in enumerate(predictions):
                target = targets_device[position].cpu().numpy()
                row = metadata.iloc[sample_index]
                condition = (int(row["snr_db"]), int(row["velocity_kmh"]))
                candidate = metrics(prediction, target)
                values = {"pi_egnn": candidate}
                for receiver, reference_rows in references.items():
                    reference = reference_rows[sample_index]
                    if (reference["snr_db"], reference["velocity_kmh"]) != condition:
                        raise ValueError(f"Condition mismatch for sample {sample_index} in {receiver} artifact.")
                    values[receiver] = reference["metrics"]
                condition_values = grouped.setdefault(
                    condition, {receiver: [] for receiver in RECEIVERS}
                )
                for receiver in RECEIVERS:
                    condition_values[receiver].append(values[receiver])
                sample_row: dict[str, object] = {
                    "sample_index": sample_index,
                    "snr_db": condition[0],
                    "velocity_kmh": condition[1],
                }
                for receiver in RECEIVERS:
                    for index, metric in enumerate(METRICS):
                        sample_row[f"{receiver}_{metric}"] = values[receiver][index]
                sample_rows.append(sample_row)
                sample_index += 1

    condition_rows = [
        condition_result(condition, values) for condition, values in sorted(grouped.items())
    ]
    expected_conditions = {
        (snr_db, velocity_kmh)
        for snr_db in (10, 15, 20)
        for velocity_kmh in (30, 120, 500)
    }
    if set(grouped) != expected_conditions or any(
        len(values["pi_egnn"]) != 15 for values in grouped.values()
    ):
        raise ValueError("Test split must contain exactly 15 samples in each of the 9 registered conditions.")
    overall_values = {
        receiver: [
            tuple(float(row[f"{receiver}_{metric}"]) for metric in METRICS)
            for row in sample_rows
        ]
        for receiver in RECEIVERS
    }
    overall = {receiver: summarize(overall_values[receiver]) for receiver in RECEIVERS}
    cache_row_by_file = {
        str(file_name): index
        for index, file_name in enumerate(feature_cache["sample_files"])
    }
    test_cache_indices = [
        cache_row_by_file[str(file_name)] for file_name in metadata["file"]
    ]
    feature_timing = {
        key: float(np.mean(feature_cache[key][test_cache_indices]))
        for key in (
            "observation_build_seconds",
            "mmse_seconds",
            "oamp_prior_seconds",
            "numpy_solve_seconds",
            "oamp_operator_setup_seconds",
            "oamp_iteration_seconds",
            "feature_compute_seconds",
        )
    }
    overall_checks = {
        f"pi_egnn_beats_{reference}_all_metrics": all(
            overall["pi_egnn"][metric] < overall[reference][metric] for metric in METRICS
        )
        for reference in ("original_gnn", "oamp_dl")
    }
    regime_a = [row for row in condition_rows if row["snr_db"] == 10]
    regime_b = [row for row in condition_rows if row["velocity_kmh"] == 120]
    regime_comparisons = {}
    for reference in ("original_gnn", "oamp_dl"):
        regime_comparisons[reference] = {
            "A_low_snr_broad": regime_result(
                "SNR 10 dB at 30, 120, and 500 km/h", regime_a, reference
            ),
            "B_120_kmh_all_snr": regime_result(
                "120 km/h at 10, 15, and 20 dB", regime_b, reference
            ),
        }
    enhancement_validity = overall_checks["pi_egnn_beats_original_gnn_all_metrics"]
    oamp_dl_bank_win = overall_checks["pi_egnn_beats_oamp_dl_all_metrics"] or any(
        regime_comparisons["oamp_dl"][regime][
            "beats_reference_all_metrics_all_conditions"
        ]
        for regime in ("A_low_snr_broad", "B_120_kmh_all_snr")
    )

    oamp_evaluation_path = (
        PROJECT_ROOT
        / config.oamp_dl.output.directory
        / config.oamp_dl.output.evaluation_file
    )
    with oamp_evaluation_path.open(encoding="utf-8") as file:
        oamp_evaluation = json.load(file)
    oamp_dl_inference_per_sample = float(
        oamp_evaluation["inference_time_per_sample_seconds"]
    )
    model_inference_per_sample = inference_seconds / len(metadata)
    dataset_and_forward_per_sample = dataset_and_forward_seconds / len(metadata)
    dataset_input_prep_per_sample = max(
        0.0, dataset_and_forward_per_sample - model_inference_per_sample
    )
    end_to_end_per_sample = (
        feature_timing["feature_compute_seconds"]
        + dataset_input_prep_per_sample
        + model_inference_per_sample
    )
    result = {
        "seed": int(config.reproducibility.seed),
        "test_samples": len(metadata),
        "prior_iterations": prior_iterations,
        "attention": "masked bidirectional softmax over active edges only",
        "edge_threshold": f"{float(config.gnn.edge_threshold_fraction_of_max):.4f} * max(abs(H_hat)) per sample",
        "ci95_method": "sample mean +/- 1.96 * sample standard deviation / sqrt(n); paired sample-level deltas",
        "overall": overall,
        "overall_success_checks": overall_checks,
        "enhancement_validity": enhancement_validity,
        "oamp_dl_bank_win_overall_or_registered_regime": oamp_dl_bank_win,
        "earns_bank_slot": enhancement_validity and oamp_dl_bank_win,
        "regimes": regime_comparisons,
        "inference_timing": {
            "prior_feature_extraction_seconds_per_sample": feature_timing,
            "dataset_loading_and_graph_packing_seconds_per_sample": dataset_input_prep_per_sample,
            "pi_egnn_model_seconds_per_sample": model_inference_per_sample,
            "pi_egnn_end_to_end_seconds_per_sample_including_prior": end_to_end_per_sample,
            "oamp_dl_seconds_per_sample_from_existing_evaluation": oamp_dl_inference_per_sample,
            "cache_used_for_features": True,
            "feature_cache_rebuild_during_evaluation": cache_built,
            "timing_note": "PI-EGNN end-to-end is the measured per-sample feature-generation time (observation extraction + MMSE + OAMP prior) plus model forward time; OAMP-DL timing is read from its existing full-test evaluation artifact.",
        },
    }
    with (output_directory / "evaluation_results.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    for filename, rows in (
        ("per_sample_results.csv", sample_rows),
        ("per_condition_results.csv", condition_rows),
    ):
        with (output_directory / filename).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()