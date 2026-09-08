"""Evaluate OAMP-DL against MMSE on the complete test split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.oamp.oamp_dl import OAMPDLDetector
from src.config.loader import load_config
from src.receivers.mmse import build_data_observation, mmse_detect
from training.oamp.dataset import load_split_metadata


def compute_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float, float]:
    prediction_bits = np.stack(
        (prediction.real >= 0, prediction.imag >= 0),
        axis=-1,
    )
    target_bits = np.stack(
        (target.real >= 0, target.imag >= 0),
        axis=-1,
    )
    bit_errors = prediction_bits != target_bits
    symbol_errors = np.any(bit_errors, axis=-1)
    signal_power = np.sum(np.abs(target) ** 2)
    error_power = np.sum(np.abs(prediction - target) ** 2)
    return (
        float(np.count_nonzero(bit_errors) / bit_errors.size),
        float(np.count_nonzero(symbol_errors) / target.size),
        float(error_power / signal_power),
    )


def summarize(
    values: list[tuple[float, float, float]],
) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    mean = array.mean(axis=0)
    standard_deviation = array.std(axis=0, ddof=1)
    confidence_interval = 1.96 * standard_deviation / np.sqrt(len(values))
    return {
        "ber": float(mean[0]),
        "ser": float(mean[1]),
        "nmse": float(mean[2]),
        "ber_std": float(standard_deviation[0]),
        "ser_std": float(standard_deviation[1]),
        "nmse_std": float(standard_deviation[2]),
        "ber_ci95": float(confidence_interval[0]),
        "ser_ci95": float(confidence_interval[1]),
        "nmse_ci95": float(confidence_interval[2]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_root = Path(config.dataset.root)
    raw_dir = dataset_root / config.dataset.raw_dir
    metadata_path = (
        dataset_root
        / config.dataset.processed_dir
        / config.dataset.split_metadata_file
    )
    metadata = load_split_metadata(metadata_path, "test")
    expected_test = int(
        config.dataset.expected_samples * config.split.test_ratio
    )
    if len(metadata) != expected_test:
        raise ValueError(
            f"Expected {expected_test} test samples; found {len(metadata)}."
        )

    observation_count = int(
        config.representation.expected_channel_shape.rows
    )
    symbol_count = int(
        config.representation.expected_channel_shape.cols
    )
    model = OAMPDLDetector(
        observation_count=observation_count,
        symbol_count=symbol_count,
        iterations=int(config.oamp_dl.iterations),
    ).to(device)
    checkpoint_path = (
        Path(config.oamp_dl.output.directory)
        / config.oamp_dl.output.checkpoint_file
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    grouped: dict[tuple[int, int], dict[str, list[tuple[float, float, float]]]] = {}
    all_mmse = []
    all_oamp_dl = []
    rows = []
    inference_seconds = 0.0

    for sample_index, row in metadata.iterrows():
        with np.load(
            raw_dir / str(row["file"]),
            allow_pickle=False,
        ) as sample:
            rx_dd = sample["rx_dd"]
            h_hat = sample["h_hat"]
            tx_dd = sample["tx_dd"]

        noise_power = 10.0 ** (-float(row["snr_db"]) / 10.0)
        y_data = build_data_observation(rx_dd, config)
        mmse_prediction = mmse_detect(
            y_data,
            h_hat,
            noise_power,
        )
        packed = np.concatenate(
            (
                y_data.reshape(-1),
                h_hat.reshape(-1),
                np.asarray([noise_power], dtype=np.complex64),
            )
        ).astype(np.complex64)
        packed_tensor = torch.from_numpy(packed).unsqueeze(0).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            oamp_dl_prediction = model(packed_tensor)[0].cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - start

        mmse_metric = compute_metrics(mmse_prediction, tx_dd)
        oamp_dl_metric = compute_metrics(oamp_dl_prediction, tx_dd)
        condition = (int(row["snr_db"]), int(row["velocity_kmh"]))
        condition_group = grouped.setdefault(
            condition,
            {"mmse": [], "oamp_dl": []},
        )
        condition_group["mmse"].append(mmse_metric)
        condition_group["oamp_dl"].append(oamp_dl_metric)
        all_mmse.append(mmse_metric)
        all_oamp_dl.append(oamp_dl_metric)
        rows.append(
            {
                "sample_index": sample_index,
                "snr_db": condition[0],
                "velocity_kmh": condition[1],
                "mmse_ber": mmse_metric[0],
                "mmse_ser": mmse_metric[1],
                "mmse_nmse": mmse_metric[2],
                "oamp_dl_ber": oamp_dl_metric[0],
                "oamp_dl_ser": oamp_dl_metric[1],
                "oamp_dl_nmse": oamp_dl_metric[2],
            }
        )

    mmse_summary = summarize(all_mmse)
    oamp_dl_summary = summarize(all_oamp_dl)
    condition_rows = []
    for (snr_db, velocity_kmh), values in sorted(grouped.items()):
        mmse_condition = summarize(values["mmse"])
        oamp_dl_condition = summarize(values["oamp_dl"])
        condition_rows.append(
            {
                "snr_db": snr_db,
                "velocity_kmh": velocity_kmh,
                "samples": len(values["mmse"]),
                "mmse_ber": mmse_condition["ber"],
                "mmse_ser": mmse_condition["ser"],
                "mmse_nmse": mmse_condition["nmse"],
                "oamp_dl_ber": oamp_dl_condition["ber"],
                "oamp_dl_ser": oamp_dl_condition["ser"],
                "oamp_dl_nmse": oamp_dl_condition["nmse"],
                "oamp_dl_beats_mmse_all_metrics": all(
                    oamp_dl_condition[metric]
                    < mmse_condition[metric]
                    for metric in ("ber", "ser", "nmse")
                ),
            }
        )

    summary = {
        "seed": int(config.reproducibility.seed),
        "test_samples": len(metadata),
        "oamp_dl_iterations": int(config.oamp_dl.iterations),
        "noise_variance_source": "per-sample configured data SNR used for H_hat generation",
        "mmse": mmse_summary,
        "oamp_dl": oamp_dl_summary,
        "inference_time_seconds": inference_seconds,
        "inference_time_per_sample_seconds": inference_seconds / len(metadata),
        "oamp_dl_beats_mmse_all_test_metrics": all(
            oamp_dl_summary[metric] < mmse_summary[metric]
            for metric in ("ber", "ser", "nmse")
        ),
        "specialist_conditions": [
            row
            for row in condition_rows
            if row["oamp_dl_beats_mmse_all_metrics"]
        ],
    }

    output_directory = Path(config.oamp_dl.output.directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    with (
        output_directory / config.oamp_dl.output.evaluation_file
    ).open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    with (
        output_directory / config.oamp_dl.output.per_sample_file
    ).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (
        output_directory / "per_condition_results.csv"
    ).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(condition_rows[0]))
        writer.writeheader()
        writer.writerows(condition_rows)

    print(json.dumps(summary, indent=2))
    print(f"Per-condition results saved to: {output_directory / 'per_condition_results.csv'}")


if __name__ == "__main__":
    main()