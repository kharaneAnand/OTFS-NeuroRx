"""Evaluate both GNN initialization ablations against OAMP-DL."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.gnn.otfs_gnn import OTFSGNN
from src.config.loader import load_config
from training.gnn.dataset import OTFSGraphDataset, load_split_metadata


def metrics(
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


def summarize(values: list[tuple[float, float, float]]) -> dict[str, float]:
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


def evaluate_initialization(
    config,
    metadata,
    raw_dir: Path,
    initialization: str,
    device: torch.device,
):
    dataset = OTFSGraphDataset(metadata, raw_dir, config, initialization)
    loader = DataLoader(
        dataset,
        batch_size=int(config.gnn.training.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = OTFSGNN(
        observation_count=int(
            config.representation.expected_channel_shape.rows
        ),
        symbol_count=int(
            config.representation.expected_channel_shape.cols
        ),
        hidden_features=int(config.gnn.hidden_features),
        layers=int(config.gnn.message_passing_layers),
    ).to(device)
    directory = (
        config.gnn.output.mmse_directory
        if initialization == "mmse"
        else config.gnn.output.neutral_directory
    )
    checkpoint = torch.load(
        Path(directory) / config.gnn.output.checkpoint_file,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_values = []
    condition_values = {}
    rows = []
    sample_index = 0

    with torch.no_grad():
        for packed, targets in loader:
            predictions = model(packed.to(device)).cpu().numpy()
            targets_np = targets.numpy()
            for position in range(predictions.shape[0]):
                row = metadata.iloc[sample_index]
                value = metrics(predictions[position], targets_np[position])
                all_values.append(value)
                condition = (int(row["snr_db"]), int(row["velocity_kmh"]))
                condition_values.setdefault(condition, []).append(value)
                rows.append(
                    {
                        "sample_index": sample_index,
                        "snr_db": condition[0],
                        "velocity_kmh": condition[1],
                        "ber": value[0],
                        "ser": value[1],
                        "nmse": value[2],
                    }
                )
                sample_index += 1

    return summarize(all_values), condition_values, rows


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

    with open(
        PROJECT_ROOT / "experiments" / "oamp_dl" / "evaluation_results.json",
        encoding="utf-8",
    ) as file:
        oamp_dl_summary = json.load(file)

    oamp_dl_conditions = {}
    oamp_dl_condition_path = (
        PROJECT_ROOT
        / "experiments"
        / "oamp_dl"
        / "per_condition_results.csv"
    )
    oamp_dl_condition_frame = pd.read_csv(
        oamp_dl_condition_path
    )
    for _, condition_row in oamp_dl_condition_frame.iterrows():
        oamp_dl_conditions[
            (int(condition_row["snr_db"]), int(condition_row["velocity_kmh"]))
        ] = {
            "ber": float(condition_row["oamp_dl_ber"]),
            "ser": float(condition_row["oamp_dl_ser"]),
            "nmse": float(condition_row["oamp_dl_nmse"]),
        }

    all_summaries = {}
    all_condition_rows = []
    all_sample_rows = []
    specialist_conditions = {}

    for initialization in ("mmse", "neutral_zero"):
        summary, condition_values, sample_rows = evaluate_initialization(
            config,
            metadata,
            raw_dir,
            initialization,
            device,
        )
        all_summaries[initialization] = summary
        for row in sample_rows:
            row["initialization"] = initialization
        all_sample_rows.extend(sample_rows)

        for condition, values in sorted(condition_values.items()):
            condition_summary = summarize(values)
            baseline = oamp_dl_conditions[condition]
            all_condition_rows.append(
                {
                    "initialization": initialization,
                    "snr_db": condition[0],
                    "velocity_kmh": condition[1],
                    "samples": len(values),
                    "gnn_ber": condition_summary["ber"],
                    "gnn_ser": condition_summary["ser"],
                    "gnn_nmse": condition_summary["nmse"],
                    "oamp_dl_ber": baseline["ber"],
                    "oamp_dl_ser": baseline["ser"],
                    "oamp_dl_nmse": baseline["nmse"],
                    "beats_oamp_dl_all_metrics": all(
                        condition_summary[metric] < baseline[metric]
                        for metric in ("ber", "ser", "nmse")
                    ),
                }
            )

    for initialization, summary in all_summaries.items():
        baseline = oamp_dl_summary["oamp_dl"]
        locked_rows = [
            row
            for row in all_condition_rows
            if row["initialization"] == initialization
            and row["snr_db"] == 10
            and row["velocity_kmh"] in {120, 500}
        ]
        specialist_conditions[initialization] = {
            "overall_beats_oamp_dl_all_metrics": all(
                summary[metric] < baseline[metric]
                for metric in ("ber", "ser", "nmse")
            ),
            "locked_regime_beats_oamp_dl_all_metrics": bool(
                locked_rows
                and all(
                    row["beats_oamp_dl_all_metrics"]
                    for row in locked_rows
                )
            ),
            "locked_regime": {
                "snr_db": [10],
                "velocity_kmh": [120, 500],
                "definition": "Pre-registered low-SNR high-mobility regime",
            },
        }

    output = {
        "seed": int(config.reproducibility.seed),
        "test_samples": len(metadata),
        "edge_threshold": "0.01 * max(abs(H_hat)) per sample",
        "message_passing_layers": int(config.gnn.message_passing_layers),
        "baseline": "OAMP-DL",
        "oamp_dl": oamp_dl_summary["oamp_dl"],
        "gnn": all_summaries,
        "success_checks": specialist_conditions,
    }

    output_root = Path("experiments")
    for initialization, directory_name in (
        ("mmse", config.gnn.output.mmse_directory),
        ("neutral_zero", config.gnn.output.neutral_directory),
    ):
        directory = Path(directory_name)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / config.gnn.output.evaluation_file).open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(output, file, indent=2)
        with (directory / config.gnn.output.per_sample_file).open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(all_sample_rows[0]),
            )
            writer.writeheader()
            writer.writerows(
                row for row in all_sample_rows
                if row["initialization"] == initialization
            )
        with (directory / config.gnn.output.per_condition_file).open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(all_condition_rows[0]),
            )
            writer.writeheader()
            writer.writerows(
                row for row in all_condition_rows
                if row["initialization"] == initialization
            )

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
