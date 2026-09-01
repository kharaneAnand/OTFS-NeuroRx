"""Evaluate the trained DNN OTFS receiver on the held-out test split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.dnn.dnn_receiver import DNNReceiver
from src.config.loader import Config, load_config
from training.dnn.dataset import (
    OTFSReceiverDataset,
    load_split_metadata,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate the trained DNN OTFS receiver."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the experiment YAML configuration.",
    )

    return parser.parse_args()


def resolve_device() -> torch.device:
    """Select the available computation device."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def build_test_dataset(
    config: Config,
) -> OTFSReceiverDataset:
    """Build the predefined test dataset."""

    dataset_root = Path(config.dataset.root)

    raw_dir = (
        dataset_root
        / config.dataset.raw_dir
    )

    metadata_path = (
        dataset_root
        / config.dataset.processed_dir
        / config.dataset.split_metadata_file
    )

    input_shape = (
        int(config.representation.expected_grid_shape.rows),
        int(config.representation.expected_grid_shape.cols),
    )

    output_symbols = int(
        config.representation.expected_data_symbols
    )

    metadata = load_split_metadata(
        metadata_path,
        "test",
    )

    return OTFSReceiverDataset(
        metadata=metadata,
        raw_dir=raw_dir,
        input_shape=input_shape,
        output_symbols=output_symbols,
    )


def build_model(config: Config) -> DNNReceiver:
    """Build the DNN using the experiment configuration."""

    input_shape = (
        int(config.representation.expected_grid_shape.rows),
        int(config.representation.expected_grid_shape.cols),
    )

    output_symbols = int(
        config.representation.expected_data_symbols
    )

    hidden_dims = [
        int(value)
        for value in config.dnn.hidden_dims
    ]

    return DNNReceiver(
        input_shape=input_shape,
        output_symbols=output_symbols,
        hidden_dims=hidden_dims,
        dropout=float(config.dnn.dropout),
    )


def load_checkpoint(
    model: DNNReceiver,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    """Load the trained model checkpoint."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"DNN checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "DNN checkpoint must contain a mapping."
        )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "DNN checkpoint does not contain model_state_dict."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


def count_parameters(
    model: torch.nn.Module,
) -> int:
    """Count trainable model parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def compute_sample_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, float, float]:
    """Compute BER, SER, and NMSE for one sample."""

    prediction_np = (
        prediction.detach()
        .cpu()
        .numpy()
    )

    target_np = (
        target.detach()
        .cpu()
        .numpy()
    )

    predicted_bits = np.stack(
        (
            prediction_np.real >= 0,
            prediction_np.imag >= 0,
        ),
        axis=-1,
    )

    target_bits = np.stack(
        (
            target_np.real >= 0,
            target_np.imag >= 0,
        ),
        axis=-1,
    )

    bit_errors = np.count_nonzero(
        predicted_bits != target_bits
    )

    total_bits = target_bits.size

    ber = (
        float(bit_errors / total_bits)
        if total_bits > 0
        else 0.0
    )

    symbol_errors = np.count_nonzero(
        np.signbit(prediction_np.real)
        != np.signbit(target_np.real)
    ) | np.count_nonzero(
        np.signbit(prediction_np.imag)
        != np.signbit(target_np.imag)
    )

    symbol_error_mask = (
        (prediction_np.real >= 0)
        != (target_np.real >= 0)
    ) | (
        (prediction_np.imag >= 0)
        != (target_np.imag >= 0)
    )

    ser = float(
        np.count_nonzero(symbol_error_mask)
        / target_np.size
    )

    signal_power = np.sum(
        np.abs(target_np) ** 2
    )

    error_power = np.sum(
        np.abs(prediction_np - target_np) ** 2
    )

    if signal_power <= 0:
        raise ValueError(
            "Target symbol power must be positive."
        )

    nmse = float(
        error_power / signal_power
    )

    return ber, ser, nmse


@torch.no_grad()
def evaluate(
    model: DNNReceiver,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Evaluate the receiver on the test set."""

    model.eval()

    total_bits = 0
    total_bit_errors = 0
    total_symbols = 0
    total_symbol_errors = 0

    total_signal_power = 0.0
    total_error_power = 0.0

    inference_time = 0.0

    per_sample_results: list[dict[str, float]] = []

    sample_index = 0

    for rx_dd, tx_dd in loader:
        rx_dd = rx_dd.to(
            device,
            non_blocking=True,
        )

        tx_dd = tx_dd.to(
            device,
            non_blocking=True,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        prediction = model(rx_dd)

        if device.type == "cuda":
            torch.cuda.synchronize()

        inference_time += (
            time.perf_counter() - start_time
        )

        if prediction.shape != tx_dd.shape:
            raise ValueError(
                "Prediction and target shapes do not match: "
                f"{prediction.shape} != {tx_dd.shape}."
            )

        prediction_np = (
            prediction.cpu().numpy()
        )

        target_np = (
            tx_dd.cpu().numpy()
        )

        predicted_real = (
            prediction_np.real >= 0
        )

        predicted_imag = (
            prediction_np.imag >= 0
        )

        target_real = (
            target_np.real >= 0
        )

        target_imag = (
            target_np.imag >= 0
        )

        bit_error_mask = (
            predicted_real != target_real
        ) | (
            predicted_imag != target_imag
        )

        symbol_error_mask = (
            bit_error_mask[..., 0]
            if bit_error_mask.ndim > 1
            else bit_error_mask
        )

        batch_bit_errors = int(
            np.count_nonzero(bit_error_mask)
        )

        batch_total_bits = int(
            bit_error_mask.size
        )

        batch_symbol_errors = int(
            np.count_nonzero(symbol_error_mask)
        )

        batch_total_symbols = int(
            target_np.size
        )

        batch_signal_power = float(
            np.sum(np.abs(target_np) ** 2)
        )

        batch_error_power = float(
            np.sum(
                np.abs(
                    prediction_np - target_np
                ) ** 2
            )
        )

        total_bit_errors += batch_bit_errors
        total_bits += batch_total_bits

        total_symbol_errors += batch_symbol_errors
        total_symbols += batch_total_symbols

        total_signal_power += batch_signal_power
        total_error_power += batch_error_power

        for batch_position in range(
            rx_dd.shape[0]
        ):
            ber, ser, nmse = compute_sample_metrics(
                prediction[batch_position],
                tx_dd[batch_position],
            )

            per_sample_results.append(
                {
                    "sample_index": float(sample_index),
                    "ber": ber,
                    "ser": ser,
                    "nmse": nmse,
                }
            )

            sample_index += 1

    if total_bits == 0 or total_symbols == 0:
        raise RuntimeError(
            "Evaluation loader contains no valid samples."
        )

    if total_signal_power <= 0:
        raise ValueError(
            "Aggregate target signal power must be positive."
        )

    metrics = {
        "ber": float(
            total_bit_errors / total_bits
        ),
        "ser": float(
            total_symbol_errors / total_symbols
        ),
        "nmse": float(
            total_error_power / total_signal_power
        ),
        "inference_time_seconds": float(
            inference_time
        ),
        "inference_time_per_sample_seconds": float(
            inference_time / sample_index
        ),
        "test_samples": float(sample_index),
    }

    return metrics, per_sample_results


def save_results(
    config: Config,
    metrics: dict[str, float],
    per_sample_results: list[dict[str, float]],
) -> None:
    """Save aggregate and per-sample evaluation results."""

    output_directory = Path(
        config.dnn.output.directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        output_directory
        / config.dnn.evaluation.output_file
    )

    per_sample_path = (
        output_directory
        / config.dnn.evaluation.per_sample_file
    )

    results = {
        "dataset_version": config.project.dataset_version,
        "model": "DNN",
        "metrics": metrics,
    }

    with results_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    import csv

    with per_sample_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_index",
                "ber",
                "ser",
                "nmse",
            ],
        )

        writer.writeheader()
        writer.writerows(
            per_sample_results
        )

    print(
        f"Evaluation results saved to: {results_path}"
    )

    print(
        f"Per-sample results saved to: {per_sample_path}"
    )


def main() -> None:
    """Run DNN evaluation."""

    args = parse_args()

    config = load_config(
        args.config
    )

    device = resolve_device()

    print(f"Device: {device}")

    dataset = build_test_dataset(
        config
    )

    loader = DataLoader(
        dataset,
        batch_size=int(
            config.dnn.training.batch_size
        ),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        config
    )

    checkpoint_path = (
        Path(config.dnn.output.directory)
        / config.dnn.output.checkpoint_file
    )

    load_checkpoint(
        model,
        checkpoint_path,
        device,
    )

    model.to(device)

    parameter_count = count_parameters(
        model
    )

    metrics, per_sample_results = evaluate(
        model=model,
        loader=loader,
        device=device,
    )

    metrics["trainable_parameters"] = float(
        parameter_count
    )

    print(
        f"Test samples: "
        f"{int(metrics['test_samples'])}"
    )

    print(
        f"BER: {metrics['ber']:.6e}"
    )

    print(
        f"SER: {metrics['ser']:.6e}"
    )

    print(
        f"NMSE: {metrics['nmse']:.6e}"
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count}"
    )

    print(
        "Inference time per sample: "
        f"{metrics['inference_time_per_sample_seconds']:.6e} s"
    )

    save_results(
        config,
        metrics,
        per_sample_results,
    )


if __name__ == "__main__":
    main()