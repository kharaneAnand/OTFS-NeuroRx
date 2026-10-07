"""Few-shot adapt only OAMP-DL scalar controls for logged ADAPT events."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.oamp.oamp_dl import OAMPDLDetector
from src.config.loader import load_config
from training.oamp.dataset import OAMPDataset

TARGET_SNR_DB = 10
TARGET_VELOCITY_KMH = 500
CALIBRATION_SPLIT = "validation"
CALIBRATION_SAMPLE_COUNT = 15
FIXED_OPTIMIZER_STEPS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
ANCHOR_COEFFICIENT = 0.1
GRADIENT_CLIP_NORM = 1.0
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    predicted_bits = np.stack((prediction.real >= 0, prediction.imag >= 0), axis=-1)
    target_bits = np.stack((target.real >= 0, target.imag >= 0), axis=-1)
    bit_errors = predicted_bits != target_bits
    symbol_errors = np.any(bit_errors, axis=-1)
    signal_power = float(np.sum(np.abs(target) ** 2))
    error_power = float(np.sum(np.abs(prediction - target) ** 2))
    return (
        float(np.count_nonzero(bit_errors) / bit_errors.size),
        float(np.count_nonzero(symbol_errors) / target.size),
        error_power / signal_power,
    )


def summarize(values: np.ndarray) -> dict[str, float]:
    means = values.mean(axis=0)
    standard_deviation = values.std(axis=0, ddof=1)
    half_width = 1.96 * standard_deviation / np.sqrt(len(values))
    return {
        metric: float(means[index])
        for index, metric in enumerate(("ber", "ser", "nmse"))
    } | {
        f"{metric}_ci95": float(half_width[index])
        for index, metric in enumerate(("ber", "ser", "nmse"))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--decision-log",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "reliability_controller" / "decision_log.csv",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not args.decision_log.is_file():
        raise FileNotFoundError(f"Controller decision log not found: {args.decision_log}")
    decision_log = pd.read_csv(args.decision_log)
    adapt_events = decision_log.loc[
        (decision_log["decision"] == "ADAPT")
        & (decision_log["recommended_adaptation_model"] == "oamp_dl")
    ].copy()
    if len(adapt_events) != 2:
        raise ValueError(
            "Expected the two saved ADAPT events recommending OAMP-DL; "
            f"found {len(adapt_events)}."
        )
    event_conditions = set(
        zip(adapt_events["snr_db"].astype(int), adapt_events["velocity_kmh"].astype(int))
    )
    expected_condition = {(TARGET_SNR_DB, TARGET_VELOCITY_KMH)}
    if event_conditions != expected_condition:
        raise ValueError(
            "Saved OAMP-DL ADAPT events do not identify the registered 10 dB / 500 km/h condition: "
            f"{sorted(event_conditions)}"
        )

    dataset_root = PROJECT_ROOT / config.dataset.root
    raw_dir = dataset_root / config.dataset.raw_dir
    metadata_path = dataset_root / config.dataset.processed_dir / config.dataset.split_metadata_file
    metadata = pd.read_csv(metadata_path)
    condition_rows = metadata.loc[
        (metadata["snr_db"].astype(int) == TARGET_SNR_DB)
        & (metadata["velocity_kmh"].astype(int) == TARGET_VELOCITY_KMH)
    ].copy()
    calibration_metadata = condition_rows.loc[
        condition_rows["split"].astype(str).str.lower() == CALIBRATION_SPLIT
    ].reset_index(drop=True)
    test_metadata = condition_rows.loc[
        condition_rows["split"].astype(str).str.lower() == "test"
    ].reset_index(drop=True)
    if len(calibration_metadata) != CALIBRATION_SAMPLE_COUNT:
        raise ValueError(
            f"Expected {CALIBRATION_SAMPLE_COUNT} calibration frames for the condition; "
            f"found {len(calibration_metadata)}."
        )
    if len(test_metadata) < 2:
        raise ValueError("Need at least two held-out test frames for paired CI95.")
    calibration_files = set(calibration_metadata["file"].astype(str))
    test_files = set(test_metadata["file"].astype(str))
    event_files = set(adapt_events["file"].astype(str))
    if calibration_files & test_files:
        raise ValueError("Calibration and test frames overlap.")
    if calibration_files & event_files:
        raise ValueError("A logged test ADAPT event was selected for calibration.")

    observation_count = int(config.representation.expected_channel_shape.rows)
    symbol_count = int(config.representation.expected_channel_shape.cols)
    model = OAMPDLDetector(
        observation_count,
        symbol_count,
        int(config.oamp_dl.iterations),
    ).to(device)
    original_checkpoint_path = (
        PROJECT_ROOT
        / config.oamp_dl.output.directory
        / config.oamp_dl.output.checkpoint_file
    )
    checkpoint = torch.load(
        original_checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    trainable_names = {"step_logits", "damping_logits", "variance_logits"}
    trainable_parameters = []
    scalar_parameter_count = 0
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in trainable_names)
        if parameter.requires_grad:
            trainable_parameters.append(parameter)
            scalar_parameter_count += parameter.numel()
    expected_parameter_count = 3 * int(config.oamp_dl.iterations)
    if scalar_parameter_count != expected_parameter_count:
        raise ValueError(
            f"Expected {expected_parameter_count} trainable OAMP-DL scalars; "
            f"found {scalar_parameter_count}."
        )
    original_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name in trainable_names
    }

    calibration_dataset = OAMPDataset(calibration_metadata, raw_dir, config)
    calibration_samples = [calibration_dataset[index] for index in range(len(calibration_dataset))]
    packed_calibration = torch.stack([sample[0] for sample in calibration_samples]).to(device)
    target_calibration = torch.stack([sample[1] for sample in calibration_samples]).to(device)

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    history = []
    for step in range(1, FIXED_OPTIMIZER_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(packed_calibration)
        data_loss = nn.functional.mse_loss(prediction.real, target_calibration.real)
        data_loss = data_loss + nn.functional.mse_loss(
            prediction.imag, target_calibration.imag
        )
        squared_drift = torch.cat(
            [
                (parameter - original_parameters[name]).reshape(-1)
                for name, parameter in model.named_parameters()
                if name in trainable_names
            ]
        ).square().mean()
        anchor_loss = ANCHOR_COEFFICIENT * squared_drift
        total_loss = data_loss + anchor_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError("Non-finite few-shot adaptation loss.")
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, GRADIENT_CLIP_NORM)
        optimizer.step()
        history.append(
            {
                "step": step,
                "data_loss": float(data_loss.detach().cpu()),
                "anchor_loss": float(anchor_loss.detach().cpu()),
                "total_loss": float(total_loss.detach().cpu()),
            }
        )
    model.eval()

    output_directory = PROJECT_ROOT / "experiments" / "oamp_dl_adapt_snr10_v500"
    output_directory.mkdir(parents=True, exist_ok=True)
    adapted_checkpoint_path = output_directory / "oamp_dl_adapted.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "base_checkpoint": str(original_checkpoint_path.relative_to(PROJECT_ROOT)),
            "adaptation": {
                "condition": {"snr_db": TARGET_SNR_DB, "velocity_kmh": TARGET_VELOCITY_KMH},
                "calibration_split": CALIBRATION_SPLIT,
                "calibration_samples": calibration_metadata["file"].astype(str).tolist(),
                "controller_adapt_events": adapt_events[
                    ["frame_index", "sample_index", "file", "snr_db", "velocity_kmh"]
                ].to_dict(orient="records"),
                "trainable_parameters": sorted(trainable_names),
                "trainable_scalar_count": scalar_parameter_count,
                "optimizer": "AdamW",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "anchor_coefficient": ANCHOR_COEFFICIENT,
                "gradient_clip_norm": GRADIENT_CLIP_NORM,
                "fixed_optimizer_steps": FIXED_OPTIMIZER_STEPS,
                "seed": SEED,
            },
        },
        adapted_checkpoint_path,
    )
    with (output_directory / "adaptation_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    test_dataset = OAMPDataset(test_metadata, raw_dir, config)
    test_samples = [test_dataset[index] for index in range(len(test_dataset))]
    packed_test = torch.stack([sample[0] for sample in test_samples]).to(device)
    target_test = torch.stack([sample[1] for sample in test_samples]).to(device)
    base_model = OAMPDLDetector(
        observation_count,
        symbol_count,
        int(config.oamp_dl.iterations),
    ).to(device)
    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model.eval()
    with torch.no_grad():
        original_predictions = base_model(packed_test).cpu().numpy()
        adapted_predictions = model(packed_test).cpu().numpy()
    targets = target_test.cpu().numpy()
    original_metrics = np.asarray(
        [sample_metrics(prediction, target) for prediction, target in zip(original_predictions, targets)],
        dtype=float,
    )
    adapted_metrics = np.asarray(
        [sample_metrics(prediction, target) for prediction, target in zip(adapted_predictions, targets)],
        dtype=float,
    )
    deltas = adapted_metrics - original_metrics
    metrics_names = ("ber", "ser", "nmse")
    comparison = {
        "trigger_source": str(args.decision_log),
        "adapt_events": adapt_events[
            ["frame_index", "sample_index", "file", "snr_db", "velocity_kmh", "recommended_adaptation_model"]
        ].to_dict(orient="records"),
        "adaptation_condition": {"snr_db": TARGET_SNR_DB, "velocity_kmh": TARGET_VELOCITY_KMH},
        "calibration_split": CALIBRATION_SPLIT,
        "calibration_samples": len(calibration_metadata),
        "calibration_files": calibration_metadata["file"].astype(str).tolist(),
        "test_samples": len(test_metadata),
        "test_files": test_metadata["file"].astype(str).tolist(),
        "calibration_test_overlap": sorted(calibration_files & test_files),
        "trainable_scalar_count": scalar_parameter_count,
        "fixed_optimizer_steps": FIXED_OPTIMIZER_STEPS,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "anchor_coefficient": ANCHOR_COEFFICIENT,
        "original_checkpoint_preserved": original_checkpoint_path.is_file(),
        "adapted_checkpoint": str(adapted_checkpoint_path.relative_to(PROJECT_ROOT)),
        "original": summarize(original_metrics),
        "adapted": summarize(adapted_metrics),
        "paired_adapted_minus_original": {
            metric: {
                "mean": float(deltas[:, index].mean()),
                "ci95": float(
                    1.96 * deltas[:, index].std(ddof=1) / np.sqrt(len(deltas))
                ),
                "ci95_low": float(
                    deltas[:, index].mean()
                    - 1.96 * deltas[:, index].std(ddof=1) / np.sqrt(len(deltas))
                ),
                "ci95_high": float(
                    deltas[:, index].mean()
                    + 1.96 * deltas[:, index].std(ddof=1) / np.sqrt(len(deltas))
                ),
            }
            for index, metric in enumerate(metrics_names)
        },
        "conclusion_caution": (
            "Suggestive single-condition experiment only: 15 labeled calibration frames and "
            f"{len(test_metadata)} held-out test frames. Test data did not update parameters or select steps."
        ),
    }
    with (output_directory / "evaluation_results.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(comparison, file, indent=2)
    per_sample = []
    for index, test_row in test_metadata.iterrows():
        per_sample.append(
            {
                "sample_index": index,
                "file": str(test_row["file"]),
                "snr_db": int(test_row["snr_db"]),
                "velocity_kmh": int(test_row["velocity_kmh"]),
                **{
                    f"original_{metric}": float(original_metrics[index, column])
                    for column, metric in enumerate(metrics_names)
                },
                **{
                    f"adapted_{metric}": float(adapted_metrics[index, column])
                    for column, metric in enumerate(metrics_names)
                },
            }
        )
    with (output_directory / "per_sample_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(per_sample[0]))
        writer.writeheader()
        writer.writerows(per_sample)

    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
